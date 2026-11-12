"""찾은 항목을 여는 동작."""
import os
import subprocess


def open_path(path: str):
    """파일은 연결된 프로그램으로 열고(실행 파일은 실행), 폴더는 탐색기로 연다."""
    if os.path.isdir(path):
        os.startfile(path)
    else:
        os.startfile(path, cwd=os.path.dirname(path))


def reveal(path: str):
    """탐색기에서 해당 항목을 선택한 상태로 연다. (윈도우 경로에는 따옴표가 들어갈 수 없다)"""
    subprocess.Popen(f'explorer /select,"{os.path.normpath(path)}"')
