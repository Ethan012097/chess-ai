@echo off
REM ===================================================================
REM  Cute Chess / Arena 等 GUI 要的是一個可執行檔，這個 .bat 就是包裝層。
REM
REM  %~dp0 是「這個 .bat 檔所在的資料夾」（結尾已含反斜線），
REM  用它而不是寫死 C:\code\chess_ai，這樣整個專案搬到別的路徑也不用改。
REM
REM  注意：不要在這裡 echo 任何東西。GUI 是用 stdout 跟引擎講 UCI 協定的，
REM  多印一行字就會讓握手失敗（規格 §2.3 的坑之二）。
REM  @echo off 就是為了這個。
REM
REM  除錯用：GUI 說「引擎載入失敗」時，把下面那行換成這個，
REM  就能把 stderr 存到檔案裡看：
REM      "%~dp0.venv\Scripts\python.exe" -m src.play --mode uci %* 2> "%~dp0engine_err.log"
REM ===================================================================

cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" -m src.play --mode uci %*
