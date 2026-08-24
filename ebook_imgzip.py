import zipfile
import os
import re
import shutil
import xml.etree.ElementTree as ET
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import traceback
import multiprocessing
from html.parser import HTMLParser
import argparse
import fitz  # PyMuPDF
import numpy as np

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.gif', '.jxl', '.avif')

class EpubImageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_title = False
        self.title = None
        self.img_srcs = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == 'title':
            self.in_title = True
        elif tag == 'img':
            for name, value in attrs:
                if name == 'src':
                    self.img_srcs.append(value)
        elif tag == 'image':
            for name, value in attrs:
                if name.lower() in ('xlink:href', 'href'):
                    self.img_srcs.append(value)

    def handle_endtag(self, tag):
        if tag.lower() == 'title':
            self.in_title = False

    def handle_data(self, data):
        if self.in_title:
            if self.title is None:
                self.title = data.strip()

def format_size(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"

def unique_filename(base_name, used_names, zero_fill=4):
    name = base_name
    count = 1
    base, ext = os.path.splitext(base_name)
    while name in used_names:
        suffix = f"_{str(count).zfill(zero_fill)}"
        name = f"{base}{suffix}{ext}"
        count += 1
    return name

def safe_filename_stem(name):
    name = str(name or '')

    # Windows 非法文件名字符
    name = re.sub(r'[\\/:*?"<>|]', '_', name)

    # 控制字符
    name = re.sub(r'[\x00-\x1f]', '_', name)

    # 合并连续的下划线
    name = re.sub(r'_+', '_', name)

    # Windows 不允许文件名以空格或 . 结尾
    name = name.strip(' .')

    return name or 'untitled'

def get_opf_path(epub_zip: zipfile.ZipFile):
    try:
        with epub_zip.open("META-INF/container.xml") as f:
            tree = ET.parse(f)
            root = tree.getroot()
            namespace = {'ns': 'urn:oasis:names:tc:opendocument:xmlns:container'}
            opf_path = root.find(".//ns:rootfile", namespace).attrib['full-path']
            return opf_path
    except Exception:
        return None

def get_opf_meta_properties(epub_zip):
    opf_path = get_opf_path(epub_zip)
    if not opf_path or opf_path not in epub_zip.namelist():
        return {}

    try:
        with epub_zip.open(opf_path) as f:
            tree = ET.parse(f)
            root = tree.getroot()
            ns = {'opf': 'http://www.idpf.org/2007/opf'}

            props = {}
            for meta in root.findall(".//opf:meta", ns):
                prop = meta.attrib.get('property', '').strip().lower()
                name = meta.attrib.get('name', '').strip().lower()
                content = meta.attrib.get('content', '').strip().lower()
                value = (meta.text or '').strip().lower()

                if prop:
                    props[prop] = value
                elif name:
                    props[name] = content
            return props
    except Exception:
        return {}

def is_manga_epub(epub_zip: zipfile.ZipFile):
    props = get_opf_meta_properties(epub_zip)
    return (
        props.get('rendition:layout') == 'pre-paginated' or
        props.get('fixed-layout') == 'true' or
        props.get('media:mediaprofile') in ['divina', 'pre-paginated']
    )

def is_cover_image(image_path, page_title, cover_images):
    if image_path in cover_images:
        return True

    stem = os.path.splitext(
        os.path.basename(image_path)
    )[0].lower()

    title = (page_title or '').strip().lower()

    cover_names = {
        'cover',
        'frontcover',
        'front_cover',
        '封面',
    }

    return stem in cover_names or title in cover_names

def get_epub_cover_images(epub_zip: zipfile.ZipFile):
    opf_path = get_opf_path(epub_zip)
    if not opf_path or opf_path not in epub_zip.namelist():
        return set()

    try:
        with epub_zip.open(opf_path) as f:
            tree = ET.parse(f)

        root = tree.getroot()
        ns = {
            'opf': 'http://www.idpf.org/2007/opf',
        }

        opf_dir = os.path.dirname(opf_path)

        manifest = {}
        cover_images = set()

        for item in root.findall('.//opf:manifest/opf:item', ns):
            item_id = item.attrib.get('id')
            href = item.attrib.get('href')
            properties = item.attrib.get('properties', '')

            if not item_id or not href:
                continue

            item_path = os.path.normpath(
                os.path.join(opf_dir, href)
            ).replace('\\', '/')

            manifest[item_id] = item_path

            # EPUB 3
            if 'cover-image' in properties.split():
                cover_images.add(item_path)

        # EPUB 2
        for meta in root.findall('.//opf:metadata/opf:meta', ns):
            if meta.attrib.get('name', '').lower() == 'cover':
                cover_id = meta.attrib.get('content')

                if cover_id and cover_id in manifest:
                    cover_images.add(manifest[cover_id])

        return cover_images

    except Exception as e:
        print(f"[警告] 无法解析 EPUB 封面信息: {e}")
        return set()

def get_spine_image_order(epub_zip: zipfile.ZipFile):
    opf_path = get_opf_path(epub_zip)
    if not opf_path or opf_path not in epub_zip.namelist():
        return []

    try:
        with epub_zip.open(opf_path) as f:
            tree = ET.parse(f)

        root = tree.getroot()
        ns = {
            'opf': 'http://www.idpf.org/2007/opf',
        }

        # manifest: id -> href
        manifest = {}

        for item in root.findall('.//opf:manifest/opf:item', ns):
            item_id = item.attrib.get('id')
            href = item.attrib.get('href')

            if item_id and href:
                item_path = os.path.normpath(
                    os.path.join(os.path.dirname(opf_path), href)
                ).replace('\\', '/')

                manifest[item_id] = item_path

        image_order = []
        seen_images = set()

        # 严格按照 spine 顺序
        for itemref in root.findall('.//opf:spine/opf:itemref', ns):
            item_idref = itemref.attrib.get('idref')
            html_file = manifest.get(item_idref)

            if not html_file or html_file not in epub_zip.namelist():
                continue

            try:
                parser = EpubImageParser()
                parser.feed(
                    epub_zip.read(html_file).decode(
                        'utf-8',
                        errors='ignore'
                    )
                )

                page_title = (
                    parser.title
                    or os.path.splitext(
                        os.path.basename(html_file)
                    )[0]
                )

                # 按 XHTML 内图片出现顺序
                for img_src in parser.img_srcs:
                    img_path = os.path.normpath(
                        os.path.join(
                            os.path.dirname(html_file),
                            img_src
                        )
                    ).replace('\\', '/')

                    if (
                        img_path.lower().endswith(IMAGE_EXTENSIONS)
                        and img_path in epub_zip.namelist()
                        and img_path not in seen_images
                    ):
                        image_order.append(
                            (img_path, page_title)
                        )
                        seen_images.add(img_path)

            except Exception as e:
                print(
                    f"[警告] 无法解析 spine 文件 "
                    f"{html_file}: {e}"
                )

        return image_order

    except Exception as e:
        print(f"[警告] 无法解析 EPUB spine: {e}")
        return []

def sample_spine_pages(spine_files, count=10):
    length = len(spine_files)

    if length <= count:
        return spine_files

    result = []
    result.extend(spine_files[:3])

    middle_start = max(0, length // 2 - 2)
    result.extend(spine_files[middle_start:middle_start + 4])

    result.extend(spine_files[-3:])

    return list(dict.fromkeys(result))


def get_spine_html_files(epub_zip):
    opf_path = get_opf_path(epub_zip)

    if not opf_path:
        return []

    try:
        with epub_zip.open(opf_path) as f:
            root = ET.parse(f).getroot()

        ns = {'opf': 'http://www.idpf.org/2007/opf'}

        manifest = {}

        for item in root.findall('.//opf:manifest/opf:item', ns):
            item_id = item.attrib.get('id')
            href = item.attrib.get('href')

            if item_id and href:
                path = os.path.normpath(
                    os.path.join(os.path.dirname(opf_path), href)
                ).replace('\\', '/')

                manifest[item_id] = path

        spine_files = []

        for itemref in root.findall('.//opf:spine/opf:itemref', ns):
            idref = itemref.attrib.get('idref')
            html_file = manifest.get(idref)

            if html_file and html_file in epub_zip.namelist():
                spine_files.append(html_file)

        return spine_files

    except Exception as e:
        print(f"[警告] 获取 spine 失败: {e}")
        return []


def has_epub_text_content(epub_zip, sample_count=10):
    try:
        spine_files = get_spine_html_files(epub_zip)

        if not spine_files:
            return False

        samples = sample_spine_pages(
            spine_files,
            sample_count
        )

        total_length = 0
        checked = 0

        for html_file in samples:
            try:
                content = epub_zip.read(html_file).decode(
                    'utf-8',
                    errors='ignore'
                )

                content = re.sub(
                    r'<(script|style).*?>.*?</\1>',
                    '',
                    content,
                    flags=re.I | re.S
                )

                text = re.sub(r'<[^>]+>', '', content)
                text = re.sub(r'\s+', ' ', text).strip()

                total_length += len(text)
                checked += 1

            except Exception:
                continue

        if checked == 0:
            return False

        average_length = total_length / checked

        print(
            f"文本检测: 采样 {checked} 页，平均字符 {average_length:.1f}"
        )

        return average_length > 300

    except Exception as e:
        print(f"[警告] 文本检测失败: {e}")
        return False

def extract_images_from_epub(epub_path, output_zip_path, skip_manga=True, delete_source=False):
    start_time = time.time()

    try:
        with zipfile.ZipFile(epub_path, 'r') as epub_zip:
            is_manga = is_manga_epub(epub_zip)

            if skip_manga and is_manga:
                return "skipped", epub_path, 0, 0

            all_files = [f for f in epub_zip.namelist() if not f.endswith('/')]

            image_files = [
                f for f in all_files
                if f.lower().endswith(IMAGE_EXTENSIONS)
            ]

            if not image_files:
                return "skipped", epub_path, 0, 0

            image_ratio = len(image_files) / len(all_files)

            if not is_manga:
                if image_ratio < 0.30:
                    print(
                        f"[{os.path.basename(epub_path)}] 图片比例 {image_ratio:.1%}，跳过"
                    )
                    return "skipped", epub_path, 0, 0

                if has_epub_text_content(epub_zip, sample_count=10):
                    print(
                        f"[{os.path.basename(epub_path)}] 检测到文本 EPUB，跳过"
                    )
                    return "skipped", epub_path, 0, 0

            image_order = get_spine_image_order(epub_zip)

            if not image_order:
                return "skipped", epub_path, 0, 0

            cover_images = get_epub_cover_images(epub_zip)

            with tempfile.TemporaryDirectory() as temp_dir:
                used_names = set()
                page_number = 1
                cover_written = False

                for image_path, page_title in image_order:
                    ext = os.path.splitext(image_path)[1].lower()

                    is_cover = (
                        image_path in cover_images
                        or is_cover_image(
                            image_path,
                            page_title,
                            cover_images
                        )
                    )

                    if is_cover and not cover_written:
                        number = 0
                        cover_written = True
                    else:
                        number = page_number
                        page_number += 1

                    safe_title = safe_filename_stem(page_title)
                    original_name = safe_filename_stem(
                        os.path.splitext(os.path.basename(image_path))[0]
                    )

                    filename = f"{number:04d}_{safe_title}_{original_name}{ext}"
                    filename = unique_filename(filename, used_names)
                    used_names.add(filename)

                    target_path = os.path.join(temp_dir, filename)

                    with epub_zip.open(image_path) as source, open(target_path, 'wb') as target:
                        shutil.copyfileobj(source, target)

                output_files = sorted(os.listdir(temp_dir))

                if not output_files:
                    return "skipped", epub_path, 0, 0

                with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as out_zip:
                    for filename in output_files:
                        out_zip.write(
                            os.path.join(temp_dir, filename),
                            arcname=filename
                        )

        size = os.path.getsize(output_zip_path)
        elapsed = time.time() - start_time

        if delete_source:
            try:
                os.remove(epub_path)
            except Exception:
                pass

        return "success", epub_path, elapsed, size
    except Exception as e:
        print(f"[错误] 处理 {epub_path} 失败：{e}")
        traceback.print_exc()
        return "failed", epub_path, 0, 0

def is_blank_page_by_pixmap(page, mean_threshold=250, std_threshold=5):
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), colorspace=fitz.csGRAY)
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        mean = np.mean(img_data)
        std = np.std(img_data)
        return mean > mean_threshold and std < std_threshold
    except Exception as e:
        print(f"[警告] 渲染页失败: {e}")
        return False

def extract_images_from_pdf(pdf_path, output_zip_path, delete_source=False):
    start_time = time.time()
    doc = None
    try:
        doc = fitz.open(pdf_path)
        image_count = 0

        with tempfile.TemporaryDirectory() as temp_dir:
            used_names = set()
            for page_index in range(len(doc)):
                page = doc[page_index]

                if is_blank_page_by_pixmap(page):
                    continue

                images = page.get_images(full=True)
                for img_index, img in enumerate(images):
                    xref = img[0]
                    base = f"{str(page_index + 1).zfill(4)}_{str(img_index + 1).zfill(2)}.jpg"
                    name = unique_filename(base, used_names)
                    used_names.add(name)
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n > 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    if pix.width < 100 or pix.height < 100:
                        continue
                    pix.save(os.path.join(temp_dir, name))
                    pix = None
                    image_count += 1

            if image_count == 0:
                return "skipped", pdf_path, 0, 0

            with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as out_zip:
                for file_name in os.listdir(temp_dir):
                    out_zip.write(os.path.join(temp_dir, file_name), arcname=file_name)

        size = os.path.getsize(output_zip_path)
        elapsed = time.time() - start_time
        doc.close()
        doc = None
        if delete_source:
            try:
                os.remove(pdf_path)
            except Exception:
                pass
        return "success", pdf_path, elapsed, size
    except Exception as e:
        print(f"[错误] 处理 {pdf_path} 失败：{e}")
        traceback.print_exc()
        return "failed", pdf_path, 0, 0
    finally:
        if doc is not None:
            doc.close()

def batch_extract(epub_root_dir, max_workers=4, skip_manga=True, delete_source=False, output_dir=None):
    if not epub_root_dir or not os.path.isdir(epub_root_dir):
        print("无效的目录路径！")
        return

    book_paths = []
    for root, _, files in os.walk(epub_root_dir):
        for file in files:
            if file.lower().endswith(('.epub', '.pdf')):
                book_paths.append(os.path.join(root, file))

    total = len(book_paths)
    if total == 0:
        print("目录及其子目录中没有找到 epub/pdf 文件")
        return

    success = 0
    skipped = 0
    failed = 0

    print(f"共找到 {total} 个文件，开始处理...\n")

    lock = Lock()
    processed_count = 0

    def process_file(path):
        ext = os.path.splitext(path)[1].lower()
        base_name = os.path.splitext(os.path.basename(path))[0] + '.zip'
        if output_dir and os.path.isdir(output_dir):
            rel_path = os.path.relpath(path, epub_root_dir)
            rel_dir = os.path.dirname(rel_path)
            out_subdir = os.path.join(output_dir, rel_dir)
            if not os.path.exists(out_subdir):
                os.makedirs(out_subdir, exist_ok=True)
            output_path = os.path.join(out_subdir, base_name)
        else:
            output_path = os.path.splitext(path)[0] + '.zip'

        if ext == '.epub':
            return extract_images_from_epub(path, output_path, skip_manga=skip_manga, delete_source=delete_source)
        elif ext == '.pdf':
            return extract_images_from_pdf(path, output_path, delete_source=delete_source)
        else:
            return "skipped", path, 0, 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, path): path for path in book_paths}

        for future in as_completed(futures):
            status, path, elapsed, size = future.result()
            name = os.path.basename(path)

            with lock:
                processed_count += 1
                print(f"[{processed_count}/{total}] 处理完毕: {name} - ", end='')
                if status == "success":
                    success += 1
                    print(f"成功，输出大小 {format_size(size)}")
                elif status == "skipped":
                    skipped += 1
                    print("跳过（漫画/无图片/非图像型 PDF）")
                else:
                    failed += 1
                    print("失败")

    print("\n====== 处理完成 ======")
    print(f"总文件数     : {total}")
    print(f"成功提取     : {success}")
    print(f"跳过         : {skipped}")
    print(f"失败         : {failed}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="提取 EPUB 和图像 PDF 中的图片并打包为 ZIP")
    parser.add_argument('-d', '--dir', type=str, help="EPUB/PDF 文件所在目录路径")
    parser.add_argument('-o', '--output-dir', type=str, default='', help="指定输出 ZIP 文件目录，留空则与原文件同目录")
    parser.add_argument('--no-skip-manga', action='store_true', help="不跳过漫画类 EPUB（即 pre-paginated）")
    parser.add_argument('--delete-source', action='store_true', help="成功提取后删除源文件")
    parser.add_argument('-w', '--workers', type=int, default=None, help="最大并发线程数（默认自动计算）")

    args = parser.parse_args()
    input_dir = args.dir
    skip_manga = not args.no_skip_manga
    delete_source = args.delete_source
    max_workers = args.workers or min(32, (multiprocessing.cpu_count() or 1) + 4)
    output_dir = args.output_dir.strip()

    if not input_dir:
        input_dir = input("请输入包含 EPUB/PDF 文件的目录路径：").strip().strip('"').replace('\\', '/')

    if not output_dir:
        output_dir_input = input("请指定输出 ZIP 文件目录（留空则与原文件同目录）：").strip().strip('"').replace('\\', '/')
        if output_dir_input:
            if not os.path.isdir(output_dir_input):
                try:
                    os.makedirs(output_dir_input, exist_ok=True)
                    print(f"已创建输出目录: {output_dir_input}")
                except Exception as e:
                    print(f"无法创建输出目录: {output_dir_input}，错误: {e}")
                    exit(1)
            output_dir = output_dir_input

    if not args.no_skip_manga:
        skip_manga_input = input("是否跳过漫画类 EPUB（y/n）？").strip().lower()
        skip_manga = (skip_manga_input != 'n')

    if not args.delete_source:
        delete_source_input = input("转换完成后是否删除原始文件（y/n）？").strip().lower()
        delete_source = (delete_source_input != 'n')

    if not input_dir or not os.path.isdir(input_dir):
        print("无效的目录路径！")
    else:
        batch_extract(
            input_dir,
            max_workers=max_workers,
            skip_manga=skip_manga,
            delete_source=delete_source,
            output_dir=output_dir if output_dir else None
        )
