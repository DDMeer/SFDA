"""PACS 数据准备与划分校验。

职责（只做这些）：
  1. 定位数据根目录，校验 4 个域齐全
  2. 校验 7 个类别与 CLASS_MAP 一致（兼容「类别名目录」与「数字目录」两种历史布局）
  3. 若数据集自带官方 kfold split 文件则直接使用，否则生成确定性分层 90/10 划分
  4. 输出可提交的划分清单到 splits/（数据本体不入库）

可选：--from-parquet 从 HuggingFace 风格 parquet 还原图片（需 pandas + pyarrow）。

用法:
    python scripts/prepare_pacs.py --check
    python scripts/prepare_pacs.py --make-splits
    python scripts/prepare_pacs.py --from-parquet <parquet 目录>
"""
import argparse
import hashlib
import json
import os
import random
import sys

sys.path.append(os.getcwd())
from models.source_model import PACS_CLASSES, PACS_DOMAINS

SPLIT_DIR = 'splits'
SEED = 2024
# 源域三分：train 训参数 / val 选 checkpoint / test 只做抗遗忘基线，绝不参与前两者
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1
SPLIT_NAMES = ('train', 'val', 'test')
IMG_EXT = ('.jpg', '.jpeg', '.png')


def find_data_root():
    """依次尝试历史上出现过的两种根目录。"""
    for cand in ('data/PACS', 'data'):
        if os.path.isdir(cand) and any(
            os.path.isdir(os.path.join(cand, d)) for d in PACS_DOMAINS
        ):
            return cand
    return None


def list_domain(root, domain):
    """返回 [(相对路径, label)]，兼容类别名目录与数字目录两种布局。

    label 一律归一化为 PACS_CLASSES 的 0-based 索引。
    """
    dom_dir = os.path.join(root, domain)
    if not os.path.isdir(dom_dir):
        raise FileNotFoundError(f'缺少域目录: {dom_dir}')

    subdirs = sorted(d for d in os.listdir(dom_dir)
                     if os.path.isdir(os.path.join(dom_dir, d)))
    if not subdirs:
        raise ValueError(f'{dom_dir} 下没有类别子目录')

    if set(subdirs) == set(PACS_CLASSES):
        layout = 'name'
        mapping = {c: i for i, c in enumerate(PACS_CLASSES)}
    elif set(subdirs) == {str(i) for i in range(len(PACS_CLASSES))}:
        layout = 'index'
        mapping = {str(i): i for i in range(len(PACS_CLASSES))}
    else:
        raise ValueError(
            f'{dom_dir} 的类别目录与 CLASS_MAP 不符。\n'
            f'  实际: {subdirs}\n'
            f'  期望(名称): {PACS_CLASSES}\n'
            f'  或(索引): {[str(i) for i in range(7)]}'
        )

    items = []
    for sub in subdirs:
        label = mapping[sub]
        d = os.path.join(dom_dir, sub)
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(IMG_EXT):
                items.append((os.path.join(domain, sub, fn), label))
    items.sort()  # 确定性
    return items, layout


def find_official_split(root, domain, kind):
    """官方 kfold 划分文件（若数据集自带）。kind: train / crossval / test"""
    for base in (root, os.path.join(root, 'splits'), 'splits_official'):
        p = os.path.join(base, f'{domain}_{kind}_kfold.txt')
        if os.path.isfile(p):
            return p
    return None


def read_official(path):
    """官方文件为 `相对路径 1-based标签`，转成 0-based。"""
    items = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rel, lab = line.rsplit(' ', 1)
            items.append((rel, int(lab) - 1))
    return items


def stratified_split(items, val_fraction=VAL_FRACTION,
                     test_fraction=TEST_FRACTION, seed=SEED):
    """确定性分层三分划分：train / val / test，按类别分组后固定种子打乱。

    用途严格互斥：
      train —— 参数训练
      val   —— checkpoint 选择
      test  —— 留出集，绝不参与训练与模型选择，用作抗遗忘基线
    """
    by_class = {}
    for rel, lab in items:
        by_class.setdefault(lab, []).append(rel)

    train, val, test = [], [], []
    for lab in sorted(by_class):
        paths = sorted(by_class[lab])                 # 先排序，消除文件系统顺序影响
        rng = random.Random(f'{seed}-{lab}')          # 每类独立且确定的随机流
        rng.shuffle(paths)
        n = len(paths)
        if n < 3:
            raise ValueError(f'类别 {lab} 只有 {n} 张，无法做三分划分')
        n_val = max(1, int(round(n * val_fraction)))
        n_test = max(1, int(round(n * test_fraction)))
        if n_val + n_test >= n:                       # 极小类别的兜底
            n_val = n_test = 1
        val.extend((p, lab) for p in paths[:n_val])
        test.extend((p, lab) for p in paths[n_val:n_val + n_test])
        train.extend((p, lab) for p in paths[n_val + n_test:])
    train.sort(); val.sort(); test.sort()
    return train, val, test


def write_split(domain, name, items):
    os.makedirs(SPLIT_DIR, exist_ok=True)
    path = os.path.join(SPLIT_DIR, f'{domain}_{name}.txt')
    with open(path, 'w') as fh:
        for rel, lab in items:
            fh.write(f'{rel} {lab}\n')
    digest = hashlib.sha256(open(path, 'rb').read()).hexdigest()[:12]
    return path, digest


def load_split(domain, name):
    """供训练/评测脚本调用。"""
    path = os.path.join(SPLIT_DIR, f'{domain}_{name}.txt')
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f'找不到划分文件 {path}，请先运行: python scripts/prepare_pacs.py --make-splits')
    items = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rel, lab = line.rsplit(' ', 1)
                items.append((rel, int(lab)))
    return items


def cmd_check(root):
    print(f'📂 数据根目录: {root}')
    print(f'🔤 类别顺序校验: {PACS_CLASSES}')
    assert PACS_CLASSES == sorted(PACS_CLASSES), 'PACS_CLASSES 必须为字母序（与官方目录序一致）'
    print('   ✅ 字母序与官方目录顺序一致')
    total = 0
    for dom in PACS_DOMAINS:
        items, layout = list_domain(root, dom)
        counts = [sum(1 for _, l in items if l == i) for i in range(len(PACS_CLASSES))]
        total += len(items)
        print(f'   {dom:<14} {len(items):>5} 张  布局={layout:<5} 每类={counts}')
    print(f'📊 合计 {total} 张（PACS 官方为 9991）')
    return total


def _official_three_way(root, dom):
    """三份官方划分齐全、且两两不相交时才采用，否则返回 None。

    PACS 官方 DG 协议里 *_test_kfold.txt 有时就是整个域（与 train 重叠），
    那种情况不能当留出集，必须退回自建划分。
    """
    paths = [find_official_split(root, dom, k) for k in ('train', 'crossval', 'test')]
    if not all(paths):
        return None
    tr, va, te = (read_official(p) for p in paths)
    s_tr, s_va, s_te = ({p for p, _ in x} for x in (tr, va, te))
    if (s_tr & s_va) or (s_tr & s_te) or (s_va & s_te):
        print(f'   ⚠️  {dom}: 官方划分存在重叠，改用确定性自建三分划分')
        return None
    return tr, va, te


def cmd_make_splits(root):
    summary = {}
    for dom in PACS_DOMAINS:
        official = _official_three_way(root, dom)
        if official is not None:
            train, val, test = official
            source = 'official_kfold'
        else:
            items, _ = list_domain(root, dom)
            train, val, test = stratified_split(items)
            tr_pct = int(round((1 - VAL_FRACTION - TEST_FRACTION) * 100))
            source = (f'stratified_{tr_pct}/{int(VAL_FRACTION*100)}/'
                      f'{int(TEST_FRACTION*100)}_seed{SEED}')
        digests = {}
        for name, part in zip(SPLIT_NAMES, (train, val, test)):
            _, digests[name] = write_split(dom, name, part)
        # 硬性保证：三份互不相交
        sets = [{p for p, _ in x} for x in (train, val, test)]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), \
            f'{dom} 的 train/val/test 存在重叠'
        summary[dom] = {'source': source, 'n_train': len(train),
                        'n_val': len(val), 'n_test': len(test),
                        'sha256': digests}
        print(f'   {dom:<14} train={len(train):>5} val={len(val):>4} '
              f'test={len(test):>4}  [{source}]')

    meta = {'seed': SEED, 'val_fraction': VAL_FRACTION, 'test_fraction': TEST_FRACTION,
            'usage': {'train': '仅参数训练', 'val': '仅 checkpoint 选择',
                      'test': '留出集；不参与训练与模型选择；抗遗忘基线'},
            'classes': PACS_CLASSES, 'domains': PACS_DOMAINS, 'splits': summary}
    with open(os.path.join(SPLIT_DIR, 'meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(f'✅ 划分与校验和已写入 {SPLIT_DIR}/（可提交，数据本体不入库）')


def cmd_from_parquet(src_dir, out_root='data'):
    """从 HuggingFace 风格 parquet 还原为 data/{域}/{类别名}/ 布局。

    schema: image: struct<bytes, path>, domain: string, label: int64(ClassLabel)
    按 row group 流式处理，避免一次性载入整个文件。
    """
    try:
        import io
        import json
        import pyarrow.parquet as pq
        from PIL import Image
    except ImportError:
        sys.exit('❌ 需要 pyarrow: pip install pyarrow')

    if os.path.isfile(src_dir) and src_dir.endswith('.parquet'):
        files = [src_dir]
    else:
        files = sorted(os.path.join(src_dir, f) for f in os.listdir(src_dir)
                       if f.endswith('.parquet'))
    if not files:
        sys.exit(f'❌ {src_dir} 下没有 .parquet 文件')

    n, skipped = 0, 0
    for fp in files:
        pf = pq.ParquetFile(fp)

        # 若文件自带 ClassLabel 名称，强制与 PACS_CLASSES 核对，不一致立即报错
        md = pf.schema_arrow.metadata or {}
        if b'huggingface' in md:
            names = (json.loads(md[b'huggingface'].decode())
                     .get('info', {}).get('features', {})
                     .get('label', {}).get('names'))
            if names and list(names) != PACS_CLASSES:
                sys.exit(f'❌ parquet 的类别顺序与 CLASS_MAP 不符\n'
                         f'   parquet: {list(names)}\n   期望   : {PACS_CLASSES}')
            if names:
                print(f'   ✅ 类别顺序与 CLASS_MAP 一致: {list(names)}')

        for g in range(pf.metadata.num_row_groups):
            tbl = pf.read_row_group(g, columns=['image', 'domain', 'label']).to_pydict()
            for im, dom_raw, lab_raw in zip(tbl['image'], tbl['domain'], tbl['label']):
                # 域与标签各自独立解析：两者都可能是「名称」或「索引」
                dom = (PACS_DOMAINS[int(dom_raw)]
                       if isinstance(dom_raw, int) else str(dom_raw))
                cls = (PACS_CLASSES[int(lab_raw)]
                       if isinstance(lab_raw, int) else str(lab_raw))
                if dom not in PACS_DOMAINS or cls not in PACS_CLASSES:
                    sys.exit(f'❌ 未知 域/类: {dom!r}/{cls!r}')

                d = os.path.join(out_root, dom, cls)
                os.makedirs(d, exist_ok=True)
                fn = os.path.basename(im.get('path') or f'{n:06d}.jpg')
                if not fn.lower().endswith(IMG_EXT):
                    fn += '.jpg'
                out_path = os.path.join(d, fn)
                if os.path.exists(out_path):
                    skipped += 1
                    continue
                Image.open(io.BytesIO(im['bytes'])).convert('RGB').save(out_path)
                n += 1
            print(f'\r   row group {g+1}/{pf.metadata.num_row_groups} '
                  f'已写 {n} 张', end='', flush=True)
        print()
    print(f'✅ 还原 {n} 张到 {out_root}/{{域}}/{{类别名}}/'
          + (f'（跳过已存在 {skipped} 张）' if skipped else ''))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='校验域/类别完整性')
    ap.add_argument('--make-splits', action='store_true', help='生成确定性 train/val 划分')
    ap.add_argument('--from-parquet', metavar='DIR', help='从 parquet 还原图片')
    args = ap.parse_args()

    if args.from_parquet:
        cmd_from_parquet(args.from_parquet)
        sys.exit(0)

    root = find_data_root()
    if root is None:
        sys.exit(
            '❌ 找不到 PACS 数据。期望以下任一布局：\n'
            '     data/{域}/{类别名}/*.jpg   或   data/{域}/{0..6}/*.jpg\n'
            '     data/PACS/{域}/...\n'
            '   若手头是 parquet：python scripts/prepare_pacs.py --from-parquet <目录>')

    if args.check or args.make_splits:
        cmd_check(root)
    if args.make_splits:
        print()
        cmd_make_splits(root)
    if not (args.check or args.make_splits):
        ap.print_help()
