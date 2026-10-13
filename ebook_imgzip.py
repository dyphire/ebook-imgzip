import zipfile
import os
import re
import shutil
import xml.etree.ElementTree as ET
import xml.sax.saxutils as saxutils
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import traceback
import multiprocessing
from html.parser import HTMLParser
import argparse
import pymupdf as fitz  # PyMuPDF
import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.gif', '.jxl', '.avif')

DEFAULT_DPI = 200
DEFAULT_QUALITY = 85

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

def parse_epub_structure(epub_zip, name_set):
    """单次解析 container.xml + OPF，返回 EPUB 结构上下文

    避免对同一份 OPF 重复 XML 解析（ manga 检测 / spine / 封面 / 元数据共用一份结果）。
    """
    opf_path = get_opf_path(epub_zip)
    if not opf_path or opf_path not in name_set:
        return None

    try:
        with epub_zip.open(opf_path) as f:
            root = ET.parse(f).getroot()
    except Exception:
        return None

    opf_dir = os.path.dirname(opf_path)

    manifest = {}   # item id -> 规范化路径
    item_props = {} # item id -> properties
    for item in root.findall(".//{*}manifest/{*}item"):
        item_id = item.attrib.get("id")
        href = item.attrib.get("href")
        if not item_id or not href:
            continue
        manifest[item_id] = os.path.normpath(os.path.join(opf_dir, href)).replace("\\", "/")
        item_props[item_id] = item.attrib.get("properties", "")

    # spine 顺序 + 阅读方向（未声明时默认 RightToLeft，适配漫画阅读）
    spine = []
    reading_direction = "RightToLeft"
    spine_elem = root.find(".//{*}spine")
    if spine_elem is not None:
        ppd = spine_elem.attrib.get("page-progression-direction", "").strip().lower()
        direction_map = {"rtl": "RightToLeft", "ltr": "LeftToRight", "vertical": "Vertical"}
        reading_direction = direction_map.get(ppd, ppd if ppd else "RightToLeft")
    for itemref in root.findall(".//{*}spine/{*}itemref"):
        idref = itemref.attrib.get("idref")
        if idref:
            spine.append(idref)

    # EPUB 3 meta property / EPUB 2 meta name-content（小写，供 manga 检测）
    meta_props = {}
    cover_id = None
    metadata = {}
    for meta in root.findall(".//{*}meta"):
        prop = meta.attrib.get("property", "").strip().lower()
        name = meta.attrib.get("name", "").strip().lower()
        content = meta.attrib.get("content", "").strip()
        value = (meta.text or "").strip().lower()
        if prop:
            meta_props[prop] = value
        elif name:
            meta_props[name] = content.lower()
        if name == "cover":
            cover_id = meta.attrib.get("content")
        elif name == "calibre:series":
            metadata["series"] = content
        elif name == "calibre:series_index":
            metadata["series_index"] = content

    # 封面图片（EPUB 3 item properties + EPUB 2 meta name=cover）
    cover_images = set()
    for item_id, props in item_props.items():
        if "cover-image" in props.split():
            cover_images.add(manifest[item_id])
    if cover_id and cover_id in manifest:
        cover_images.add(manifest[cover_id])

    # DC 元数据（保留原始大小写，供 ComicInfo.xml 使用）
    for tag in ("title", "publisher", "language", "description", "date"):
        el = root.find(f".//{{*}}{tag}")
        if el is not None and el.text and el.text.strip():
            metadata[tag] = el.text.strip()
    creators = [c.text.strip() for c in root.findall(".//{*}creator") if c.text and c.text.strip()]
    if creators:
        metadata["creators"] = creators

    spine_files = [manifest[i] for i in spine if i in manifest and manifest[i] in name_set]

    return {
        "opf_path": opf_path,
        "manifest": manifest,
        "meta_props": meta_props,
        "cover_images": cover_images,
        "metadata": metadata,
        "reading_direction": reading_direction,
        "spine_files": spine_files,
    }

def is_manga_props(meta_props):
    return (
        meta_props.get('rendition:layout') == 'pre-paginated' or
        meta_props.get('fixed-layout') == 'true' or
        meta_props.get('media:mediaprofile') in ['divina', 'pre-paginated']
    )

def build_comicinfo_xml(metadata, reading_direction):
    """根据 EPUB 元数据生成 ComicInfo.xml（兼容 Komga/Kavita 等漫画库）"""
    def esc(val):
        return saxutils.escape(str(val)) if val else ''

    year, month, day = '', '', ''
    if metadata.get('date'):
        parts = str(metadata['date']).split('-')
        if len(parts) > 0: year = parts[0]
        if len(parts) > 1: month = parts[1]
        if len(parts) > 2: day = parts[2]

    xml = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<ComicInfo>'
    ]
    if metadata.get('title'): xml.append(f"  <Title>{esc(metadata['title'])}</Title>")
    if metadata.get('series'): xml.append(f"  <Series>{esc(metadata['series'])}</Series>")
    if metadata.get('series_index'): xml.append(f"  <Number>{esc(metadata['series_index'])}</Number>")
    creators = metadata.get('creators') or []
    if creators: xml.append(f"  <Writer>{esc(', '.join(creators))}</Writer>")
    if metadata.get('publisher'): xml.append(f"  <Publisher>{esc(metadata['publisher'])}</Publisher>")
    if metadata.get('language'): xml.append(f"  <LanguageISO>{esc(metadata['language'])}</LanguageISO>")
    if metadata.get('description'): xml.append(f"  <Summary>{esc(metadata['description'])}</Summary>")
    if year: xml.append(f"  <Year>{esc(year)}</Year>")
    if month: xml.append(f"  <Month>{esc(month)}</Month>")
    if day: xml.append(f"  <Day>{esc(day)}</Day>")

    xml.append(f"  <ReadingDirection>{esc(reading_direction)}</ReadingDirection>")
    xml.append('</ComicInfo>')
    return '\n'.join(xml)

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

def get_spine_image_order(epub_zip, spine_files, name_set):
    image_order = []
    seen_images = set()

    # 严格按照 spine 顺序
    for html_file in spine_files:
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
                    and img_path in name_set
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


def has_epub_text_content(epub_zip, spine_files, sample_count=10):
    try:
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
            name_set = set(epub_zip.namelist())

            ctx = parse_epub_structure(epub_zip, name_set)
            if not ctx:
                return "skipped", epub_path, 0, 0

            is_manga = is_manga_props(ctx["meta_props"])

            if skip_manga and is_manga:
                return "skipped", epub_path, 0, 0

            all_files = [f for f in name_set if not f.endswith('/')]

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

                if has_epub_text_content(epub_zip, ctx["spine_files"], sample_count=10):
                    print(
                        f"[{os.path.basename(epub_path)}] 检测到文本 EPUB，跳过"
                    )
                    return "skipped", epub_path, 0, 0

            image_order = get_spine_image_order(epub_zip, ctx["spine_files"], name_set)

            if not image_order:
                return "skipped", epub_path, 0, 0

            cover_images = ctx["cover_images"]

            used_names = set()
            page_number = 1
            cover_written = False
            entries = []  # (压缩包内文件名, EPUB 内图片路径)

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

                entries.append((filename, image_path))

            if not entries:
                return "skipped", epub_path, 0, 0

            comicinfo_xml = build_comicinfo_xml(
                ctx["metadata"], ctx["reading_direction"]
            )

            # 直接从 EPUB 流式写入 ZIP，不经过临时目录（I/O 减半）
            pbar = tqdm(total=len(entries), desc=f"打包 {os.path.basename(epub_path)}", unit="张", leave=False) if HAS_TQDM else None
            try:
                with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as out_zip:
                    for filename, image_path in entries:
                        zinfo = zipfile.ZipInfo(filename)
                        # 已压缩格式（JPEG/PNG 等）直接存储，避免无意义的 DEFLATE CPU 开销
                        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.gif', '.avif', '.jxl')):
                            zinfo.compress_type = zipfile.ZIP_STORED
                        with epub_zip.open(image_path) as source, out_zip.open(zinfo, 'w') as target:
                            shutil.copyfileobj(source, target)
                        if pbar is not None:
                            pbar.update(1)
                    out_zip.writestr("ComicInfo.xml", comicinfo_xml)
            finally:
                if pbar is not None:
                    pbar.close()

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

def _is_blank_page(page):
    """空白页检测（低分辨率灰度渲染）"""
    blank_pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), colorspace=fitz.csGRAY)
    img_data = np.frombuffer(blank_pix.samples, dtype=np.uint8).reshape(blank_pix.height, blank_pix.width)
    return np.mean(img_data) > 250 and np.std(img_data) < 5

def _render_page_to_jpeg(page, output_file, dpi, quality):
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    img.save(output_file, "JPEG", quality=quality, optimize=True, progressive=True)

def extract_pdf_page_images(pdf_path, page_index, output_dir, padding_width, dpi, quality):
    """提取单页内嵌图像（进程池 worker）

    优先原样提取（无损、快）：JPEG 直接取出原始字节流，其余格式解码后保存为 JPEG。
    以下情况回退为该页整页渲染：带蒙版（SMask）、色彩空间缺失、ImageMask、
    带 alpha 通道、JPEG 保存失败。
    返回该页生成的文件列表（空白页 / 无可用图像的页面返回空列表）。
    """
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_index)

        if _is_blank_page(page):
            return []

        page_num = str(page_index + 1).zfill(padding_width)
        output_files = []

        for img_index, img in enumerate(page.get_images(full=True), start=1):
            xref, smask = img[0], img[1]
            img_width, img_height = img[2], img[3]
            colorspace = img[5] if len(img) > 5 else None
            filter_name = img[8] if len(img) > 8 else ''

            if img_width < 100 or img_height < 100:
                continue

            # 蒙版图像直接提取结果残缺；色彩空间缺失时无法正常解码，均整页渲染兜底
            if smask != 0 or not colorspace:
                output_file = os.path.join(output_dir, f"page_{page_num}.jpg")
                _render_page_to_jpeg(page, output_file, dpi, quality)
                return [output_file]

            # 模板蒙版（ImageMask，常为 1-bit 线条/文字镂刻）必须结合页面绘制上下文，整页渲染兜底
            try:
                if doc.xref_get_key(xref, 'ImageMask')[1] == 'true':
                    output_file = os.path.join(output_dir, f"page_{page_num}.jpg")
                    _render_page_to_jpeg(page, output_file, dpi, quality)
                    return [output_file]
            except Exception:
                pass

            # JPEG（DCTDecode）直接取出原始字节流，无损且免解码重编码
            if filter_name == 'DCTDecode':
                output_file = os.path.join(output_dir, f"page_{page_num}_img_{img_index:02d}.jpg")
                with open(output_file, 'wb') as f:
                    f.write(doc.xref_stream_raw(xref))
                output_files.append(output_file)
                continue

            pix = None
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                if pix.width < 100 or pix.height < 100:
                    continue
                if pix.alpha:
                    # 带 alpha 通道：直接提取无法保证与页面背景的正确合成，整页渲染兜底
                    pix = None
                    output_file = os.path.join(output_dir, f"page_{page_num}.jpg")
                    _render_page_to_jpeg(page, output_file, dpi, quality)
                    return [output_file]
                output_file = os.path.join(output_dir, f"page_{page_num}_img_{img_index:02d}.jpg")
                try:
                    pix.save(output_file, output="jpeg", jpg_quality=quality)
                except Exception:
                    # JPEG 保存失败（如异常的色彩空间），整页渲染兜底
                    pix = None
                    output_file = os.path.join(output_dir, f"page_{page_num}.jpg")
                    _render_page_to_jpeg(page, output_file, dpi, quality)
                    return [output_file]
                output_files.append(output_file)
            finally:
                pix = None

        return output_files
    finally:
        doc.close()

def extract_images_from_pdf(pdf_path, output_zip_path, dpi=DEFAULT_DPI, quality=DEFAULT_QUALITY, delete_source=False, render_workers=None):
    start_time = time.time()
    doc = None
    try:
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        padding_width = len(str(num_pages))

        # 文本型 PDF 检测（只读页面元数据，不渲染）：
        # 图像页占比低于 30% 视为文本型 PDF，整本跳过
        pages_with_images = 0
        for i in range(num_pages):
            if doc[i].get_images(full=True):
                pages_with_images += 1
        image_ratio = pages_with_images / num_pages if num_pages else 0
        if image_ratio < 0.30:
            print(f"[{os.path.basename(pdf_path)}] 非图像型 PDF（图像页占比 {image_ratio:.1%}），跳过")
            return "skipped", pdf_path, 0, 0
        doc.close()
        doc = None

        with tempfile.TemporaryDirectory() as temp_dir:
            jobs = []
            for page_index in range(num_pages):
                jobs.append((str(pdf_path), page_index, temp_dir, padding_width, dpi, quality))

            if not jobs:
                return "skipped", pdf_path, 0, 0

            # 参考 pdf2cbz.py：进程池并行处理各页（渲染与 optimize 编码均受 GIL 限制，必须用进程而非线程）
            if render_workers is None:
                render_workers = (multiprocessing.cpu_count() or 1)
            max_render_workers = max(1, min(len(jobs), render_workers))
            from concurrent.futures import ProcessPoolExecutor

            pdf_name = os.path.basename(pdf_path)
            pbar = tqdm(total=num_pages, desc=f"处理 {pdf_name}", unit="页", leave=False) if HAS_TQDM else None
            try:
                with ProcessPoolExecutor(max_workers=max_render_workers) as render_executor:
                    futures = {
                        render_executor.submit(extract_pdf_page_images, *job): job[1]
                        for job in jobs
                    }

                    rendered_files = []
                    for future in as_completed(futures):
                        page_index = futures[future]
                        try:
                            rendered_files.extend(future.result())
                        except Exception as e:
                            print(f"[警告] 处理第 {page_index + 1} 页失败: {e}")
                        if pbar is not None:
                            pbar.update(1)
            finally:
                if pbar is not None:
                    pbar.close()

            if not rendered_files:
                return "skipped", pdf_path, 0, 0

            # 文件名已补零，字典序即页序
            rendered_files.sort()
            pbar = tqdm(total=len(rendered_files), desc=f"打包 {pdf_name}", unit="页", leave=False) if HAS_TQDM else None
            try:
                # JPEG/PNG 已是压缩格式，直接存储（ZIP_STORED），避免 DEFLATE 的无效 CPU 开销
                with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_STORED) as out_zip:
                    for page_num, file_path in enumerate(rendered_files, start=1):
                        ext = os.path.splitext(file_path)[1].lower()
                        arcname = f"{page_num:0{padding_width}d}{ext}"
                        out_zip.write(file_path, arcname=arcname)
                        if pbar is not None:
                            pbar.update(1)
            finally:
                if pbar is not None:
                    pbar.close()

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
    finally:
        if doc is not None:
            doc.close()

def batch_extract(epub_root_dir, max_workers=4, skip_manga=True, delete_source=False, output_dir=None, dpi=DEFAULT_DPI, quality=DEFAULT_QUALITY):
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

    # 批量模式下多个文件并发处理，每个 PDF 内部的渲染进程数需相应减少，避免进程超订
    pdf_render_workers = max(1, (multiprocessing.cpu_count() or 1) // max_workers)

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
            return extract_images_from_pdf(path, output_path, dpi=dpi, quality=quality, delete_source=delete_source, render_workers=pdf_render_workers)
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
    parser.add_argument('--dpi', type=int, default=DEFAULT_DPI, help=f"PDF 渲染分辨率 DPI（默认: {DEFAULT_DPI}）")
    parser.add_argument('--quality', type=int, default=DEFAULT_QUALITY, help=f"PDF 渲染 JPEG 质量 1-100（默认: {DEFAULT_QUALITY}）")

    args = parser.parse_args()
    input_dir = args.dir
    skip_manga = not args.no_skip_manga
    delete_source = args.delete_source
    max_workers = args.workers or min(32, (multiprocessing.cpu_count() or 1) + 4)
    output_dir = args.output_dir.strip()
    dpi = args.dpi
    quality = args.quality

    if quality < 1 or quality > 100:
        print("JPEG 质量必须在 1-100 之间！")
        exit(1)

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
            output_dir=output_dir if output_dir else None,
            dpi=dpi,
            quality=quality
        )
