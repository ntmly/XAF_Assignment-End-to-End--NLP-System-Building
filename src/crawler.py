import os
import json
import argparse
import requests


class WikiCrawler:
    def __init__(self, output_dir="data/raw"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (RAG-Crawler)"
        })

    # =========================
    # GET FULL WIKI TEXT (NO FILTER)
    # =========================
    def get_full_text(self, title: str) -> str:
        url = "https://vi.wikipedia.org/w/api.php"

        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "titles": title,
            "explaintext": True,
            "redirects": 1,
            "formatversion": 2
        }

        try:
            r = self.session.get(url, params=params, timeout=10)

            print(f"[DEBUG] URL: {r.url}")
            print(f"[DEBUG] STATUS: {r.status_code}")

            if r.status_code != 200:
                return ""

            data = r.json()

            pages = data.get("query", {}).get("pages", [])
            if not pages:
                return ""

            return pages[0].get("extract", "")

        except Exception as e:
            print(f"[ERROR] {title}: {e}")
            return ""

    # =========================
    # UET + VNU ONLY
    # =========================
    def crawl_vnu_uet(self):
        docs = []

        vnu_title = "Đại_học_Quốc_gia_Hà_Nội"
        uet_title = "Trường_Đại_học_Công_nghệ,_Đại_học_Quốc_gia_Hà_Nội"

        vnu_text = self.get_full_text(vnu_title)
        uet_text = self.get_full_text(uet_title)

        docs.append("Vietnam National University, Hanoi (VNU): " + vnu_text)
        docs.append("University of Engineering and Technology (UET): " + uet_text)

        return docs

    # =========================
    # SAVE
    # =========================
    def save(self, docs):
        path = os.path.join(self.output_dir, "raw_docs.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)

        print(f"\nSaved {len(docs)} docs → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw")
    args = parser.parse_args()

    crawler = WikiCrawler(args.output)

    docs = crawler.crawl_vnu_uet()

    crawler.save(docs)

    for i, d in enumerate(docs):
        print(f"\n--- Document {i+1} ---")
        print(d[:2000])


if __name__ == "__main__":
    main()