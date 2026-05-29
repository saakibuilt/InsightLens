#!/usr/bin/env python3
from __future__ import annotations

import asyncio, math, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote, urlsplit, urlunsplit, quote

try:
    import requests
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests

try:
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright"])
    subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError


BASE = "https://www.justice.gov"
HOME = f"{BASE}/epstein"
SEARCH_URL = f"{BASE}/epstein/search"

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw_pdfs"

DATASET_COUNT = 12
MAX_PER_DATASET = 20

EXTS = (".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff")


def fixed_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((
        parts.scheme,
        parts.netloc,
        quote(unquote(parts.path), safe="/%"),
        parts.query,
        parts.fragment,
    ))


def is_file_url(url: str) -> bool:
    return urlparse(url).path.lower().endswith(EXTS)


def clean_filename(url: str) -> str:
    name = unquote(os.path.basename(urlparse(url).path))
    return re.sub(r"[^A-Za-z0-9._ -]+", "_", name)


def valid_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1000:
        return False

    head = path.read_bytes()[:16]
    return (
        head.startswith(b"%PDF")
        or head.startswith(b"\xff\xd8\xff")
        or head.startswith(b"\x89PNG")
        or head.startswith(b"II*\x00")
        or head.startswith(b"MM\x00*")
    )


def parse_counts_from_text(text: str):
    text = " ".join(text.split())

    m = re.search(
        r"Showing\s+(\d[\d,]*)\s+to\s+(\d[\d,]*)\s+of\s+(\d[\d,]*)\s+results",
        text,
        re.I,
    )
    if m:
        start = int(m.group(1).replace(",", ""))
        end = int(m.group(2).replace(",", ""))
        total = int(m.group(3).replace(",", ""))
        per_page = max(1, end - start + 1)
        pages = math.ceil(total / per_page)
        return start, end, total, pages, per_page

    m = re.search(r"(\d[\d,]*)\s+results?", text, re.I)
    if m:
        total = int(m.group(1).replace(",", ""))
        return None, None, total, None, None

    return None, None, None, None, None


async def page_text(page) -> str:
    try:
        return await page.locator("body").inner_text(timeout=15000)
    except Exception:
        return ""


async def parse_counts(page):
    return parse_counts_from_text(await page_text(page))


async def click_yes(page):
    for sel in ["button:has-text('Yes')", "a:has-text('Yes')", "input[value='Yes']", "text=Yes"]:
        try:
            loc = page.locator(sel).first
            if await loc.count():
                await loc.click(timeout=5000)
                await page.wait_for_timeout(1000)
                return
        except Exception:
            pass


async def wait_ready(page, timeout_ms: int = 60000):
    await page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)

    try:
        await page.wait_for_load_state("networkidle", timeout=timeout_ms)
    except Exception:
        pass

    try:
        await page.wait_for_function(
            """
            () => {
                const txt = document.body ? document.body.innerText : "";
                const hasCount =
                    /Showing\\s+\\d[\\d,]*\\s+to\\s+\\d[\\d,]*\\s+of\\s+\\d[\\d,]*\\s+results/i.test(txt)
                    || /\\d[\\d,]*\\s+results?/i.test(txt);

                const hasFile = [...document.querySelectorAll("a[href]")]
                    .some(a => /\\.(pdf|jpe?g|png|tiff?)(\\?|#|$)/i.test(a.href));

                return hasCount || hasFile;
            }
            """,
            timeout=timeout_ms,
        )
    except Exception:
        pass

    await page.wait_for_timeout(700)


async def extract_file_links(page) -> list[str]:
    hrefs = await page.locator("a[href]").evaluate_all(
        """
        els => els
            .map(a => a.href)
            .filter(h => /\\.(pdf|jpe?g|png|tiff?)(\\?|#|$)/i.test(h))
        """
    )

    out = []
    for href in hrefs:
        href = fixed_url(href)
        if is_file_url(href) and href not in out:
            out.append(href)

    return out


async def wait_for_range(page, expected_start: int, timeout_ms: int = 90000):
    try:
        await page.wait_for_function(
            """
            (expectedStart) => {
                const txt = document.body ? document.body.innerText : "";
                const m = txt.match(/Showing\\s+(\\d[\\d,]*)\\s+to\\s+(\\d[\\d,]*)\\s+of\\s+(\\d[\\d,]*)\\s+results/i);
                if (!m) return false;

                const start = parseInt(m[1].replaceAll(",", ""), 10);

                const hasFile = [...document.querySelectorAll("a[href]")]
                    .some(a => /\\.(pdf|jpe?g|png|tiff?)(\\?|#|$)/i.test(a.href));

                return start === expectedStart && hasFile;
            }
            """,
            arg=expected_start,
            timeout=timeout_ms,
        )
    except PlaywrightTimeoutError:
        await page.wait_for_timeout(3000)


async def click_page_number(page, target_page: int, expected_start: int) -> bool:
    clicked = await page.evaluate(
        """
        ({targetPage}) => {
            const wanted = String(targetPage);

            function visible(el) {
                const r = el.getBoundingClientRect();
                const s = window.getComputedStyle(el);
                return r.width > 0 && r.height > 0 && s.visibility !== "hidden" && s.display !== "none";
            }

            const all = [...document.querySelectorAll("a, button, [role='button'], [tabindex]")];

            const candidates = all.filter(el => {
                if (!visible(el)) return false;

                const txt = (el.innerText || el.textContent || "").trim();
                const aria = (el.getAttribute("aria-label") || "").trim().toLowerCase();
                const title = (el.getAttribute("title") || "").trim().toLowerCase();

                return (
                    txt === wanted ||
                    aria === `page ${wanted}` ||
                    aria === `go to page ${wanted}` ||
                    title === `page ${wanted}` ||
                    title === `go to page ${wanted}`
                );
            });

            const el = candidates[candidates.length - 1];
            if (!el) return false;

            el.scrollIntoView({block: "center", inline: "center"});
            el.click();
            return true;
        }
        """,
        {"targetPage": target_page},
    )

    if not clicked:
        return False

    await wait_for_range(page, expected_start)
    return True


async def click_next_button(page, expected_start: int) -> bool:
    clicked = await page.evaluate(
        """
        () => {
            function visible(el) {
                const r = el.getBoundingClientRect();
                const s = window.getComputedStyle(el);
                return r.width > 0 && r.height > 0 && s.visibility !== "hidden" && s.display !== "none";
            }

            const all = [...document.querySelectorAll("a, button, [role='button'], [tabindex]")];

            const candidates = all.filter(el => {
                if (!visible(el)) return false;

                const txt = (el.innerText || el.textContent || "").trim().toLowerCase();
                const aria = (el.getAttribute("aria-label") || "").trim().toLowerCase();
                const title = (el.getAttribute("title") || "").trim().toLowerCase();

                return (
                    txt === "next" ||
                    txt === "next ›" ||
                    txt === "›" ||
                    txt === "»" ||
                    aria.includes("next") ||
                    title.includes("next")
                );
            });

            const el = candidates[candidates.length - 1];
            if (!el) return false;

            el.scrollIntoView({block: "center", inline: "center"});
            el.click();
            return true;
        }
        """
    )

    if not clicked:
        return False

    await wait_for_range(page, expected_start)
    return True


async def go_to_page(page, target_page: int, per_page: int) -> bool:
    expected_start = ((target_page - 1) * per_page) + 1

    if await click_page_number(page, target_page, expected_start):
        return True

    if await click_next_button(page, expected_start):
        return True

    return False


async def go_search_and_submit(page, term: str):
    await page.goto(HOME, wait_until="domcontentloaded", timeout=90000)
    await page.wait_for_timeout(1000)
    await click_yes(page)

    try:
        await page.locator("a:has-text('Search Full Library')").first.click(timeout=8000)
        await page.wait_for_timeout(1500)
    except Exception:
        await page.goto(SEARCH_URL, wait_until="domcontentloaded", timeout=90000)

    await page.evaluate(
        """
        (term) => {
            const input =
                [...document.querySelectorAll("input, textarea")]
                    .find(i => (i.placeholder || "").toLowerCase().includes("type to search"))
                || document.querySelector("input[type='search'], input");

            if (!input) throw new Error("Search input not found");

            input.focus();
            input.value = term;
            input.dispatchEvent(new Event("input", {bubbles:true}));
            input.dispatchEvent(new Event("change", {bubbles:true}));

            const btn = [...document.querySelectorAll("button,input[type=submit],a")]
                .find(b => ((b.innerText || b.value || "").trim().toLowerCase() === "search"));

            if (!btn) throw new Error("Search button not found");

            btn.click();
        }
        """,
        term,
    )

    await wait_ready(page)
    await wait_for_range(page, 1)


async def make_requests_session(context) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Referer": SEARCH_URL,
        "Accept": "application/pdf,image/*,*/*",
    })

    for c in await context.cookies():
        s.cookies.set(
            c["name"],
            c["value"],
            domain=c.get("domain"),
            path=c.get("path", "/"),
        )

    return s


def download_stream(session: requests.Session, url: str, folder: Path) -> bool:
    folder.mkdir(parents=True, exist_ok=True)
    filename = clean_filename(url)
    dest = folder / filename

    if valid_file(dest):
        print(f"    exists: {filename}")
        return True

    print(f"    downloading: {filename}")

    try:
        with session.get(url, stream=True, timeout=120, allow_redirects=True) as r:
            if r.status_code != 200:
                print(f"    failed HTTP {r.status_code}: {filename}")
                return False

            with open(dest, "wb") as f:
                for chunk in r.iter_content(1024 * 256):
                    if chunk:
                        f.write(chunk)

        if valid_file(dest):
            return True

        dest.unlink(missing_ok=True)
        print(f"    invalid/HTML response: {filename}")
        return False

    except Exception as e:
        print(f"    failed: {filename} -> {e}")
        return False


async def scrape_and_download_pages(
    page,
    context,
    folder: Path,
    max_links: int | None = None,
) -> tuple[int, int]:
    session = await make_requests_session(context)

    start, end, total, pages, per_page = await parse_counts(page)

    if not per_page:
        per_page = 10

    if total and not pages:
        pages = math.ceil(total / per_page)

    print(f"  site total: {total or '?'} | total pages: {pages or '?'} | per page: {per_page or '?'}")

    seen_links = set()
    total_found = 0
    total_kept = 0
    page_no = 1

    while True:
        await wait_ready(page)

        start, end, total_now, pages_now, per_page_now = await parse_counts(page)
        if total_now:
            total = total or total_now
        if pages_now:
            pages = pages or pages_now
        if per_page_now:
            per_page = per_page_now

        links = await extract_file_links(page)

        page_links = []
        for href in links:
            if href not in seen_links:
                seen_links.add(href)
                page_links.append(href)

                if max_links and total_found + len(page_links) >= max_links:
                    break

        print(
            f"  page {page_no}/{pages or '?'}: found {len(page_links)} new files | "
            f"running found: {len(seen_links)}"
        )

        for link in page_links:
            if max_links and total_found >= max_links:
                break

            total_found += 1
            if download_stream(session, link, folder):
                total_kept += 1

        print(f"  page {page_no}: downloaded/kept running {total_kept}/{total_found}")

        if max_links and total_found >= max_links:
            break

        if pages and page_no >= pages:
            break

        next_page = page_no + 1

        moved = await go_to_page(page, next_page, per_page)
        if not moved:
            print(f"  could not move to page {next_page}; stopping")
            break

        page_no = next_page

    return total_kept, total_found


async def run_search(page, context, term: str):
    print(f"\nSearching full Epstein Library for: {term}")
    await go_search_and_submit(page, term)

    folder = OUT_DIR / "search_results" / re.sub(r"[^A-Za-z0-9._-]+", "_", term)
    kept, found = await scrape_and_download_pages(page, context, folder)

    print(f"\nDownloaded/kept {kept}/{found} search result files")


async def collect_dataset_links(page, context, dataset_url: str, folder: Path):
    await page.goto(dataset_url, wait_until="domcontentloaded", timeout=90000)
    await click_yes(page)
    await wait_ready(page)

    return await scrape_and_download_pages(
        page,
        context,
        folder,
        max_links=MAX_PER_DATASET,
    )


async def run_datasets(page, context):
    print(f"Found {DATASET_COUNT} datasets")

    for i in range(1, DATASET_COUNT + 1):
        dataset_url = f"{BASE}/epstein/doj-disclosures/data-set-{i}-files"
        folder = OUT_DIR / f"data_set_{i}"

        print(f"\nData Set {i}: {dataset_url}")
        kept, found = await collect_dataset_links(page, context, dataset_url, folder)

        print(f"  downloaded/kept {kept}/{found}")


async def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    term = " ".join(sys.argv[1:]).strip()
    tmp_profile = tempfile.mkdtemp(prefix="doj_epstein_browser_")

    try:
        async with async_playwright() as p:
            context = await p.chromium.launch_persistent_context(
                user_data_dir=tmp_profile,
                headless=False,
                accept_downloads=True,
                args=["--no-first-run", "--no-default-browser-check"],
            )

            page = context.pages[0] if context.pages else await context.new_page()
            page.set_default_timeout(60000)

            if term:
                await run_search(page, context, term)
            else:
                await run_datasets(page, context)

            await context.close()

    finally:
        shutil.rmtree(tmp_profile, ignore_errors=True)

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())