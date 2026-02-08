@echo off
chcp 65001 > nul
cd /d "%~dp0"

echo 正在初始化 Git 仓库...
git init

echo 正在重命名分支为 main...
git branch -M main

echo 正在添加文件...
git add .

echo 正在提交更改...
git commit -m "Initial commit"

echo.
echo ========================================================
echo [成功] 本地仓库已初始化并提交。
echo.
echo 下一步：
echo 1. 在 GitHub 上创建一个新的空仓库 (不要勾选初始化 README 等)。
echo 2. 复制仓库的 URL (例如 https://github.com/username/repo.git)。
echo 3. 在此窗口运行以下命令 (将 URL 替换为你的):
echo.
echo    git remote add origin <你的仓库URL>
echo    git push -u origin main
echo.
echo ========================================================
cmd /k

