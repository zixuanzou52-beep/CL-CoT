import argparse
import csv
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _read_json(path: Path) -> object:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: object, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _resolve_dir(candidates: Sequence[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(" / ".join(str(p) for p in candidates))


def _download_file(url: str, dest_path: Path, *, overwrite: bool) -> Path:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and not overwrite:
        return dest_path
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with urllib.request.urlopen(url) as resp, open(tmp_path, "wb") as f:
        shutil.copyfileobj(resp, f)
    tmp_path.replace(dest_path)
    return dest_path


def _extract_zip(zip_path: Path, dest_dir: Path, *, overwrite: bool) -> Path:
    if dest_dir.exists() and overwrite:
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    return dest_dir


def _find_single_subdir(parent: Path) -> Path:
    subdirs = [p for p in parent.iterdir() if p.is_dir()]
    if len(subdirs) != 1:
        raise RuntimeError(f"Expected exactly 1 directory under {parent}, found {len(subdirs)}")
    return subdirs[0]


def _ensure_github_repo(
    *,
    repo: str,
    ref: str,
    dest_dir: Path,
    raw_dir: Path,
    overwrite: bool,
    download: bool,
) -> Path:
    dest_dir = dest_dir.resolve()
    if dest_dir.exists() and any(dest_dir.iterdir()):
        return dest_dir
    if not download:
        raise FileNotFoundError(f"Missing local dataset directory: {dest_dir}")

    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_name = f"{repo.replace('/', '_')}-{ref}.zip"
    zip_path = raw_dir / zip_name
    if ref.startswith("v") or ref[0].isdigit():
        url = f"https://github.com/{repo}/archive/refs/tags/{ref}.zip"
    else:
        url = f"https://github.com/{repo}/archive/refs/heads/{ref}.zip"
    _download_file(url, zip_path, overwrite=overwrite)

    tmp_extract = raw_dir / f"_tmp_extract_{repo.replace('/', '_')}"
    if tmp_extract.exists():
        shutil.rmtree(tmp_extract)
    _extract_zip(zip_path, tmp_extract, overwrite=True)
    repo_root = _find_single_subdir(tmp_extract)

    if dest_dir.exists() and overwrite:
        shutil.rmtree(dest_dir)
    if not dest_dir.exists():
        shutil.move(str(repo_root), str(dest_dir))
    shutil.rmtree(tmp_extract)
    return dest_dir


def _read_tsv_dicts(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [{k: (v or "") for k, v in row.items()} for row in reader]


def _read_csv_table(path: Path) -> Tuple[List[str], List[List[str]]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = [r for r in csv.reader(f)]
    if not rows:
        return [], []
    headers = [str(x) for x in rows[0]]
    body = [[str(x) for x in r] for r in rows[1:]]
    return headers, body


def _normalize_tabfact_label(label: object) -> str:
    if isinstance(label, bool):
        return "ENTAILED" if label else "REFUTED"
    if isinstance(label, int):
        return "ENTAILED" if label == 1 else "REFUTED"
    s = str(label).strip().lower()
    if s in {"entailed", "entail", "supported", "support", "true", "1"}:
        return "ENTAILED"
    if s in {"refuted", "refute", "contradicted", "false", "0"}:
        return "REFUTED"
    return str(label).strip()


def _ensure_split_files(
    *,
    root: Path,
    names: Sequence[str],
    split: str,
) -> Path:
    for name in names:
        p = root / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing {split} split file under {root}: {', '.join(names)}")


def prepare_wtq(
    *,
    output_dir: Path,
    wtq_root: Path,
    raw_dir: Path,
    overwrite: bool,
    download: bool,
    train_tsv: str,
    dev_tsv: str,
    test_tsv: str,
) -> None:
    wtq_root = _ensure_github_repo(
        repo="ppasupat/WikiTableQuestions",
        ref="v1.0.2",
        dest_dir=wtq_root,
        raw_dir=raw_dir,
        overwrite=overwrite,
        download=download,
    )
    split_paths = {
        "train": wtq_root / train_tsv,
        "dev": wtq_root / dev_tsv,
        "test": wtq_root / test_tsv,
    }
    missing = [k for k, p in split_paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"WTQ split files missing under {wtq_root}: {missing}")

    out_base = output_dir / "wtq"
    for split, tsv_path in split_paths.items():
        processed: List[Dict[str, object]] = []
        for ex in _read_tsv_dicts(tsv_path):
            rel = ex.get("context", "")
            if not rel:
                continue
            table_path = (wtq_root / rel).resolve()
            if not table_path.exists() and table_path.with_suffix(".csv").exists():
                table_path = table_path.with_suffix(".csv")
            if not table_path.exists():
                continue
            headers, rows = _read_csv_table(table_path)
            processed.append(
                {
                    "table": {"headers": headers, "rows": rows},
                    "question": ex.get("utterance", ""),
                    "answer": ex.get("targetValue", ""),
                    "table_id": rel,
                    "example_id": ex.get("id", ""),
                }
            )
        _write_json(out_base / f"{split}.json", processed, overwrite=overwrite)


def _tabfact_iter_examples(obj: object) -> Iterable[Tuple[str, str, object, Optional[str]]]:
    if isinstance(obj, dict):
        for table_id, payload in obj.items():
            if isinstance(payload, list) and len(payload) >= 2 and isinstance(payload[0], list) and isinstance(payload[1], list):
                statements = payload[0]
                labels = payload[1]
                caption = payload[2] if len(payload) >= 3 else None
                for stmt, lbl in zip(statements, labels):
                    yield str(table_id), str(stmt), lbl, str(caption) if caption is not None else None
            elif isinstance(payload, dict):
                stmt = payload.get("statement") or payload.get("sentence")
                lbl = payload.get("label")
                caption = payload.get("caption") or payload.get("table_caption")
                if stmt is not None and lbl is not None:
                    yield str(table_id), str(stmt), lbl, str(caption) if caption is not None else None
    elif isinstance(obj, list):
        for item in obj:
            if not isinstance(item, dict):
                continue
            table_id = item.get("table_id") or item.get("tableId") or item.get("id") or item.get("table")
            stmt = item.get("statement") or item.get("sentence") or item.get("question")
            lbl = item.get("label")
            caption = item.get("caption") or item.get("table_caption")
            if table_id is None or stmt is None or lbl is None:
                continue
            yield str(table_id), str(stmt), lbl, str(caption) if caption is not None else None


def prepare_tabfact(
    *,
    output_dir: Path,
    tabfact_root: Path,
    raw_dir: Path,
    overwrite: bool,
    download: bool,
    ref: str,
) -> None:
    tabfact_root = _ensure_github_repo(
        repo="wenhuchen/Table-Fact-Checking",
        ref=ref,
        dest_dir=tabfact_root,
        raw_dir=raw_dir,
        overwrite=overwrite,
        download=download,
    )
    tables_dir = _resolve_dir([tabfact_root / "data" / "all_csv", tabfact_root / "all_csv"])
    splits_dir = _resolve_dir([tabfact_root / "tokenized_data", tabfact_root / "data", tabfact_root])

    split_files = {
        "train": _ensure_split_files(root=splits_dir, names=("train.json", "train_data.json"), split="train"),
        "dev": _ensure_split_files(root=splits_dir, names=("val.json", "valid.json", "dev.json"), split="dev"),
        "test": _ensure_split_files(root=splits_dir, names=("test.json",), split="test"),
    }

    out_base = output_dir / "tabfact"
    for split, path in split_files.items():
        obj = _read_json(path)
        processed: List[Dict[str, object]] = []
        for i, (table_id, stmt, lbl, caption) in enumerate(_tabfact_iter_examples(obj)):
            table_path = tables_dir / f"{table_id}.csv"
            if not table_path.exists():
                continue
            headers, rows = _read_csv_table(table_path)
            label = _normalize_tabfact_label(lbl)
            ex: Dict[str, object] = {
                "table": {"headers": headers, "rows": rows},
                "question": stmt,
                "answer": label,
                "label": label,
                "table_id": table_id,
                "example_id": f"tabfact_{table_id}_{i}",
            }
            if caption:
                ex["table_caption"] = caption
            processed.append(ex)
        _write_json(out_base / f"{split}.json", processed, overwrite=overwrite)


def _hybridqa_table_to_headers_rows(table_obj: object) -> Tuple[List[str], List[List[str]]]:
    if isinstance(table_obj, dict):
        if "headers" in table_obj and "rows" in table_obj:
            return (
                [str(x) for x in table_obj.get("headers", [])],
                [[str(c) for c in r] for r in table_obj.get("rows", [])],
            )
        header = table_obj.get("header")
        data = table_obj.get("data") or table_obj.get("rows")
        if isinstance(header, list) and isinstance(data, list) and data and isinstance(data[0], list):
            return [str(x) for x in header], [[str(c) for c in r] for r in data]
    return [], []


def _hybridqa_extract_context(ex: Dict[str, object]) -> str:
    parts: List[str] = []
    for k in ("intro", "section_title", "section_text"):
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    return "\n\n".join(parts)


def prepare_hybridqa(
    *,
    output_dir: Path,
    hybridqa_root: Path,
    raw_dir: Path,
    overwrite: bool,
    download: bool,
    ref: str,
    wikitables_ref: str,
) -> None:
    hybridqa_root = _ensure_github_repo(
        repo="wenhuchen/HybridQA",
        ref=ref,
        dest_dir=hybridqa_root,
        raw_dir=raw_dir,
        overwrite=overwrite,
        download=download,
    )
    splits_dir = _resolve_dir([hybridqa_root / "released_data", hybridqa_root / "data", hybridqa_root])
    split_files = {
        "train": splits_dir / "train.json",
        "dev": splits_dir / "dev.json",
        "test": splits_dir / "test.json",
    }
    missing = [k for k, p in split_files.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"HybridQA split files missing under {splits_dir}: {missing}")

    out_base = output_dir / "hybridqa"
    for split, path in split_files.items():
        obj = _read_json(path)
        if isinstance(obj, dict) and isinstance(obj.get("data"), list):
            items = obj["data"]
        elif isinstance(obj, list):
            items = obj
        else:
            raise ValueError(f"Unsupported HybridQA split format: {path}")

        need_lookup = any(isinstance(ex, dict) and ex.get("table") is None for ex in items)
        table_lookup: Optional[Dict[str, object]] = None
        if need_lookup:
            wikitables_root = _ensure_github_repo(
                repo="wenhuchen/WikiTables-WithLinks",
                ref=wikitables_ref,
                dest_dir=raw_dir / "WikiTables-WithLinks",
                raw_dir=raw_dir,
                overwrite=overwrite,
                download=download,
            )
            candidates = [
                wikitables_root / "tables.jsonl",
                wikitables_root / "tables_with_links.jsonl",
                wikitables_root / "data" / "tables.jsonl",
                wikitables_root / "data" / "tables_with_links.jsonl",
            ]
            table_file = next((p for p in candidates if p.exists()), None)
            if table_file is None:
                jsonl_candidates = sorted([p for p in wikitables_root.rglob("*.jsonl") if "table" in p.name.lower()])
                if not jsonl_candidates:
                    raise FileNotFoundError(f"Could not find table jsonl under {wikitables_root}")
                table_file = jsonl_candidates[0]
            table_lookup = {}
            with open(table_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        t = json.loads(line)
                    except Exception:
                        continue
                    uid = t.get("uid") or t.get("table_id") or t.get("id")
                    if uid is not None:
                        table_lookup[str(uid)] = t

        processed: List[Dict[str, object]] = []
        for i, ex in enumerate(items):
            if not isinstance(ex, dict):
                continue
            question = ex.get("question") or ex.get("query") or ex.get("utterance")
            answer = ex.get("answer_text") or ex.get("answer") or ex.get("answers")
            table_id = ex.get("table_id") or ex.get("uid") or ex.get("tableId")
            table_obj = ex.get("table")
            if table_obj is None and table_lookup is not None and table_id is not None:
                table_obj = table_lookup.get(str(table_id))
            if question is None or table_id is None:
                continue
            headers, rows = _hybridqa_table_to_headers_rows(table_obj)
            answer_text = " ; ".join(str(x) for x in answer) if isinstance(answer, list) else ("" if answer is None else str(answer))
            processed.append(
                {
                    "table": {"headers": headers, "rows": rows},
                    "question": str(question),
                    "answer": answer_text,
                    "table_id": str(table_id),
                    "example_id": ex.get("question_id") or ex.get("id") or f"hybridqa_{table_id}_{i}",
                    "context": _hybridqa_extract_context(ex),
                }
            )
        _write_json(out_base / f"{split}.json", processed, overwrite=overwrite)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare datasets for CL-CoT")
    parser.add_argument("--dataset", type=str, default="wtq", choices=["wtq", "tabfact", "hybridqa", "all"])
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--no_download", action="store_true")
    parser.add_argument("--wtq_root", type=str, default=None)
    parser.add_argument("--tabfact_root", type=str, default=None)
    parser.add_argument("--hybridqa_root", type=str, default=None)
    parser.add_argument("--wtq_train", type=str, default="data/training.tsv")
    parser.add_argument("--wtq_dev", type=str, default="data/random-split-seed-1-test.tsv")
    parser.add_argument("--wtq_test", type=str, default="data/pristine-unseen-tables.tsv")
    parser.add_argument("--tabfact_ref", type=str, default="master")
    parser.add_argument("--hybridqa_ref", type=str, default="master")
    parser.add_argument("--wikitables_ref", type=str, default="master")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    raw_dir = Path(args.raw_dir)
    overwrite = bool(args.overwrite)
    download = not bool(args.no_download)

    if args.dataset in {"wtq", "all"}:
        wtq_root = Path(args.wtq_root) if args.wtq_root else raw_dir / "WikiTableQuestions-v1.0.2"
        prepare_wtq(
            output_dir=output_dir,
            wtq_root=wtq_root,
            raw_dir=raw_dir,
            overwrite=overwrite,
            download=download,
            train_tsv=str(args.wtq_train),
            dev_tsv=str(args.wtq_dev),
            test_tsv=str(args.wtq_test),
        )

    if args.dataset in {"tabfact", "all"}:
        tabfact_root = Path(args.tabfact_root) if args.tabfact_root else raw_dir / "Table-Fact-Checking"
        prepare_tabfact(
            output_dir=output_dir,
            tabfact_root=tabfact_root,
            raw_dir=raw_dir,
            overwrite=overwrite,
            download=download,
            ref=str(args.tabfact_ref),
        )

    if args.dataset in {"hybridqa", "all"}:
        hybridqa_root = Path(args.hybridqa_root) if args.hybridqa_root else raw_dir / "HybridQA"
        prepare_hybridqa(
            output_dir=output_dir,
            hybridqa_root=hybridqa_root,
            raw_dir=raw_dir,
            overwrite=overwrite,
            download=download,
            ref=str(args.hybridqa_ref),
            wikitables_ref=str(args.wikitables_ref),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

