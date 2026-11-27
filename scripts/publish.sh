#!/usr/bin/env bash
# 把本工程（代码 + 两段演示视频）提交到考核仓库。
#
#   ./scripts/publish.sh
#
# 需要：
#   1. 已在本目录跑过 run_assessment.sh，存在 results/demo_A.mp4 与 results/demo_B.mp4
#   2. 有该仓库的写权限，并准备好 GitHub Personal Access Token（见文末提示）
#
# 脚本会新建 $HOME/work/submit 作为暂存目录，只复制需要提交的文件，
# 绝不会把 data/（1 GB 视频）、build/、results/ 带进仓库。
set -euo pipefail

cd "$(dirname "$0")/.."
SRC="$(pwd)"

REPO_URL="https://github.com/Minggogogo-byt/bingbingbing.git"
STAGE="$HOME/work/submit"
AUTHOR_NAME="${GIT_AUTHOR_NAME:-张铭权}"
AUTHOR_MAIL="${GIT_AUTHOR_EMAIL:-zhangmingquan@example.com}"

echo "=== 1/6 检查必要文件 ==="
for f in CMakeLists.txt README.md .gitignore src include config; do
  [[ -e "$f" ]] || { echo "缺少 $f，请在工程根目录运行" >&2; exit 2; }
done
for v in results/demo_A.mp4 results/demo_B.mp4; do
  if [[ ! -f "$v" ]]; then
    echo "缺少 $v —— 先跑一遍：" >&2
    echo "  ffmpeg -y -i results/demo_A.avi -c:v libx264 -crf 23 results/demo_A.mp4" >&2
    exit 2
  fi
  sz=$(stat -c%s "$v")
  printf "  %-24s %6.1f MB\n" "$v" "$(echo "scale=2; $sz/1048576" | bc)"
  if (( sz > 95 * 1024 * 1024 )); then
    echo "  ↑ 超过 95 MB，GitHub 会拒绝。压缩一下再跑：" >&2
    echo "    ffmpeg -y -i $v -vf scale=960:720 -crf 28 ${v%.mp4}_small.mp4" >&2
    exit 2
  fi
done

echo
echo "=== 2/6 准备暂存目录 $STAGE ==="
rm -rf "$STAGE"
if git clone --depth 1 "$REPO_URL" "$STAGE" 2>/dev/null; then
  echo "  已克隆远程仓库（保留其原有内容与历史）"
else
  echo "  克隆失败（仓库为空或无权限读取），改为新建本地仓库"
  mkdir -p "$STAGE"
  git -C "$STAGE" init -q
  git -C "$STAGE" remote add origin "$REPO_URL"
fi

echo
echo "=== 3/6 复制需要提交的文件 ==="
cp -r "$SRC"/src "$SRC"/include "$SRC"/config "$SRC"/scripts "$SRC"/docs "$STAGE"/ 2>/dev/null || true
cp "$SRC"/CMakeLists.txt "$SRC"/README.md "$SRC"/.gitignore "$STAGE"/
mkdir -p "$STAGE/media"
cp "$SRC"/results/demo_A.mp4 "$STAGE/media/demo_A.mp4"
cp "$SRC"/results/demo_B.mp4 "$STAGE/media/demo_B.mp4"
echo "  完成"

echo
echo "=== 4/6 确认没有大文件混进来 ==="
find "$STAGE" -path "$STAGE/.git" -prune -o -type f -size +95M -print | while read -r big; do
  echo "  警告：$big 超过 95 MB" >&2
done
echo "  最大的 6 个文件："
find "$STAGE" -path "$STAGE/.git" -prune -o -type f -printf '%s\t%p\n' 2>/dev/null \
  | sort -rn | head -6 | awk -F'\t' '{printf "  %7.1f MB  %s\n", $1/1048576, $2}'

echo
echo "=== 5/6 提交 ==="
cd "$STAGE"
git config user.name "$AUTHOR_NAME"
git config user.email "$AUTHOR_MAIL"
git add -A
git status --short | head -40
echo "  ---"
git diff --cached --stat | tail -3
git commit -q -m "26智科张铭权：装甲板检测与 PnP 位姿估计（含两段演示视频）" \
  || echo "  没有改动需要提交（可能已提交过）"

echo
echo "=== 6/6 推送 ==="
BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)"
echo "  分支：$BRANCH"
echo "  接下来会要求输入 GitHub 用户名和 token（密码位置粘贴 token）"
git push -u origin "$BRANCH"

echo
echo "完成。仓库地址：${REPO_URL%.git}"
echo "提示：GitHub 已不支持账号密码推送，密码处要粘贴 Personal Access Token。"
echo "     生成方法：GitHub → Settings → Developer settings → Personal access tokens"
echo "              → Tokens (classic) → Generate new token → 勾选 repo → 复制保存"
