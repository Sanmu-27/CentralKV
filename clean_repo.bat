@echo off
chcp 65001 > nul
cd /d "%~dp0"

echo ========================================================
echo 正在清理 GitHub 上的多余文件...
echo ========================================================

echo.
echo [1/3] 删除本地的脚本文件...
REM 删除我们自己生成的工具脚本
del setup_git.bat
del upload_to_github.bat

echo.
echo [2/3] 提交删除操作...
git add .
git commit -m "Remove utility scripts"

echo.
echo [3/3] 推送更改到 GitHub...
git push origin main

echo.
echo ========================================================
echo [完成] 清理完毕！请刷新 GitHub 页面。
echo ========================================================
echo 这个窗口可以关闭了。
pause
del clean_repo.bat

