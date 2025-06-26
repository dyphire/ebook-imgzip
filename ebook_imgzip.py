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

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.gif')

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
                value = (meta.text or '').strip().lower()
                if prop:
                    props[prop] = value
            return props
    except Exception:
        return {}

def is_manga_epub(epub_zip: zipfile.ZipFile):
    props = get_opf_meta_properties(epub_zip)
    media_profile = props.get('media:mediaprofile', '')
    rendition_layout = props.get('rendition:layout', '')
    return rendition_layout == 'pre-paginated' or media_profile in ['divina', 'pre-paginated']

def format_size(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"

def unique_filename(base_name, used_names):
    name = base_name
    count = 1
    while name in used_names:
        name = f"{os.path.splitext(base_name)[0]}_{count}{os.path.splitext(base_name)[1]}"
        count += 1
    return name

def safe_path_component(name):
    name = re.sub(r'[\\/:*?"<>|]', '_', name)
    name = name.strip(" .")
    return name

def extract_images_from_epub(epub_path, output_zip_path, skip_manga=True, delete_source=False):
    start_time = time.time()
    try:
        with zipfile.ZipFile(epub_path, 'r') as epub_zip:
            if skip_manga and is_manga_epub(epub_zip):
                return "skipped", epub_path, 0, 0

            all_files = [f for f in epub_zip.namelist() if not f.endswith('/')]
            total_files = len(all_files)
            image_files = [f for f in all_files if f.lower().endswith(IMAGE_EXTENSIONS)]

            if total_files == 0 or not image_files:
                return "skipped", epub_path, 0, 0

            image_ratio = len(image_files) / total_files
            if image_ratio < 0.3:
                print(f"[{os.path.basename(epub_path)}] 图片比例仅 {image_ratio:.1%}，低于 30%，跳过提取")
                return "skipped", epub_path, 0, 0

            img_to_title = {}
            html_files = [f for f in all_files if f.lower().endswith(('.html', '.xhtml'))]
            for html_file in html_files:
                try:
                    html_bytes = epub_zip.read(html_file)
                    parser = EpubImageParser()
                    parser.feed(html_bytes.decode('utf-8', errors='ignore'))
                    page_title = parser.title or os.path.splitext(os.path.basename(html_file))[0]
                    for img_src in parser.img_srcs:
                        img_path = os.path.normpath(os.path.join(os.path.dirname(html_file), img_src)).replace('\\', '/')
                        if img_path in image_files:
                            img_to_title[img_path] = page_title
                except Exception:
                    pass

            parent_dirs = set(os.path.basename(os.path.dirname(p)) or 'root' for p in image_files)
            use_subdir = len(parent_dirs) > 1

            with tempfile.TemporaryDirectory() as temp_dir:
                used_names = set()
                fallback_index = 1
                for image_file in image_files:
                    base_ext = os.path.splitext(image_file)[1].lower()
                    page_title = img_to_title.get(image_file, os.path.splitext(os.path.basename(image_file))[0])
                    safe_title = "".join(c if c.isalnum() or c in " _-()" else "_" for c in page_title) or "untitled"

                    lower_title = safe_title.lower()
                    is_cover = lower_title in ("cover", "封面")
                    match = re.search(r'\d+', safe_title)
                    if is_cover:
                        number_prefix = "0000"
                    elif match:
                        number_prefix = match.group(0).zfill(4)
                    else:
                        number_prefix = str(9000 + fallback_index).zfill(4)
                        fallback_index += 1

                    base_name = f"{number_prefix}_{safe_title}{base_ext}"
                    name = unique_filename(base_name, used_names)
                    used_names.add(name)

                    parent_base = os.path.basename(os.path.dirname(image_file))
                    parent_dir_name = safe_path_component(parent_base) if parent_base else None

                    if is_cover or not use_subdir or not parent_dir_name:
                        target_subdir = temp_dir
                    else:
                        target_subdir = os.path.join(temp_dir, parent_dir_name)
                    os.makedirs(target_subdir, exist_ok=True)

                    target_path = os.path.join(target_subdir, name)
                    with epub_zip.open(image_file) as source, open(target_path, 'wb') as target:
                        shutil.copyfileobj(source, target)

                with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as out_zip:
                    for root, _, files in os.walk(temp_dir):
                        for file in files:
                            abs_path = os.path.join(root, file)
                            rel_path = os.path.relpath(abs_path, temp_dir)
                            out_zip.write(abs_path, arcname=rel_path)

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
