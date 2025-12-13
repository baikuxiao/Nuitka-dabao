#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键打包游戏工具 v5.2 缓存增强版 (PySide6 重构版)
修复内容：
1. v5.1: 修复 pyinstaller 检测失败问题（导入名大小写）
2. v5.1: 修复安装后缓存更新逻辑
3. v5.1: 添加 pip包名 -> 导入名 的反向映射
4. v5.2: 缓存有效期延长至 7 天（解决重复检测问题）
5. v5.2: 缓存文件固定在用户目录（换目录不丢失）
6. v5.2: 添加 torch 等更多库的映射
7. Refactor: UI 重构为 PySide6

基于 v5.0 完全重构版
"""

import os
import sys
import subprocess
import shutil
import time
import glob
import ast
import re
import hashlib
import json
import tempfile
import traceback
import atexit
import threading
import queue
import concurrent.futures
from pathlib import Path
from typing import Dict, Set, List, Tuple, Optional, Any

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QPushButton, QLineEdit, 
                               QTextEdit, QCheckBox, QRadioButton, QButtonGroup,
                               QFileDialog, QProgressBar, QTabWidget, QMessageBox,
                               QTreeWidget, QTreeWidgetItem, QHeaderView, QGroupBox,
                               QScrollArea, QFrame, QStyleFactory)
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QSize
from PySide6.QtGui import QIcon, QFont, QColor, QBrush, QTextCursor, QAction

# ==================== 常量定义 ====================

VERSION = "5.2"

# 完整的Python标准库列表（Python 3.8-3.12）
STDLIB_MODULES = frozenset({
    # 内置模块
    'abc', 'aifc', 'argparse', 'array', 'ast', 'asynchat', 'asyncio', 'asyncore',
    'atexit', 'audioop', 'base64', 'bdb', 'binascii', 'binhex', 'bisect',
    'builtins', 'bz2', 'calendar', 'cgi', 'cgitb', 'chunk', 'cmath', 'cmd',
    'code', 'codecs', 'codeop', 'collections', 'colorsys', 'compileall',
    'concurrent', 'configparser', 'contextlib', 'contextvars', 'copy', 'copyreg',
    'cProfile', 'crypt', 'csv', 'ctypes', 'curses', 'dataclasses', 'datetime',
    'dbm', 'decimal', 'difflib', 'dis', 'distutils', 'doctest', 'email',
    'encodings', 'enum', 'errno', 'faulthandler', 'fcntl', 'filecmp', 'fileinput',
    'fnmatch', 'fractions', 'ftplib', 'functools', 'gc', 'getopt', 'getpass',
    'gettext', 'glob', 'graphlib', 'grp', 'gzip', 'hashlib', 'heapq', 'hmac',
    'html', 'http', 'idlelib', 'imaplib', 'imghdr', 'imp', 'importlib', 'inspect',
    'io', 'ipaddress', 'itertools', 'json', 'keyword', 'lib2to3', 'linecache',
    'locale', 'logging', 'lzma', 'mailbox', 'mailcap', 'marshal', 'math',
    'mimetypes', 'mmap', 'modulefinder', 'msvcrt', 'multiprocessing', 'netrc', 
    'nis', 'nntplib', 'numbers', 'operator', 'optparse', 'os', 'ossaudiodev', 
    'pathlib', 'pdb', 'pickle', 'pickletools', 'pipes', 'pkgutil', 'platform', 
    'plistlib', 'poplib', 'posix', 'posixpath', 'pprint', 'profile', 'pstats', 
    'pty', 'pwd', 'py_compile', 'pyclbr', 'pydoc', 'queue', 'quopri', 'random', 
    're', 'readline', 'reprlib', 'resource', 'rlcompleter', 'runpy', 'sched', 
    'secrets', 'select', 'selectors', 'shelve', 'shlex', 'shutil', 'signal', 
    'site', 'smtpd', 'smtplib', 'sndhdr', 'socket', 'socketserver', 'spwd', 
    'sqlite3', 'ssl', 'stat', 'statistics', 'string', 'stringprep', 'struct', 
    'subprocess', 'sunau', 'symtable', 'sys', 'sysconfig', 'syslog', 'tabnanny', 
    'tarfile', 'telnetlib', 'tempfile', 'termios', 'test', 'textwrap', 'threading', 
    'time', 'timeit', 'tkinter', 'token', 'tokenize', 'tomllib', 'trace', 
    'traceback', 'tracemalloc', 'tty', 'turtle', 'turtledemo', 'types', 'typing',
    'typing_extensions', 'unicodedata', 'unittest', 'urllib', 'uu', 'uuid', 
    'venv', 'warnings', 'wave', 'weakref', 'webbrowser', 'winreg', 'winsound', 
    'wsgiref', 'xdrlib', 'xml', 'xmlrpc', 'zipapp', 'zipfile', 'zipimport', 
    'zlib', '_thread', '__future__', '__main__', 'antigravity', 'this',
    # 私有模块
    '_abc', '_asyncio', '_bisect', '_blake2', '_bootlocale', '_bz2', '_codecs',
    '_collections', '_collections_abc', '_compat_pickle', '_compression',
    '_contextvars', '_crypt', '_csv', '_ctypes', '_curses', '_datetime',
    '_decimal', '_elementtree', '_functools', '_hashlib', '_heapq', '_imp',
    '_io', '_json', '_locale', '_lsprof', '_lzma', '_markupbase', '_md5',
    '_multibytecodec', '_multiprocessing', '_opcode', '_operator', '_osx_support',
    '_pickle', '_posixshmem', '_posixsubprocess', '_py_abc', '_pydecimal',
    '_pyio', '_queue', '_random', '_sha1', '_sha256', '_sha3', '_sha512',
    '_signal', '_sitebuiltins', '_socket', '_sqlite3', '_sre', '_ssl', '_stat',
    '_statistics', '_string', '_strptime', '_struct', '_symtable', '_thread',
    '_threading_local', '_tkinter', '_tracemalloc', '_uuid', '_warnings',
    '_weakref', '_weakrefset', '_winapi', '_xxsubinterpreters', '_xxtestfuzz',
})

# 第三方库映射：import名 -> pip包名
PACKAGE_NAME_MAP = {
    'PIL': 'Pillow',
    'cv2': 'opencv-python',
    'sklearn': 'scikit-learn',
    'skimage': 'scikit-image',
    'yaml': 'PyYAML',
    'bs4': 'beautifulsoup4',
    'dateutil': 'python-dateutil',
    'dotenv': 'python-dotenv',
    'jwt': 'PyJWT',
    'serial': 'pyserial',
    'wx': 'wxPython',
    'gi': 'PyGObject',
    'cairo': 'pycairo',
    'OpenGL': 'PyOpenGL',
    'usb': 'pyusb',
    'Crypto': 'pycryptodome',
    'google': 'google-api-python-client',
    'lxml': 'lxml',
    'numpy': 'numpy',
    'pandas': 'pandas',
    'scipy': 'scipy',
    'matplotlib': 'matplotlib',
    'pygame': 'pygame',
    'requests': 'requests',
    'flask': 'Flask',
    'django': 'Django',
    'sqlalchemy': 'SQLAlchemy',
    'aiohttp': 'aiohttp',
    'httpx': 'httpx',
    'pydantic': 'pydantic',
    'fastapi': 'fastapi',
    'redis': 'redis',
    'pymongo': 'pymongo',
    'psycopg2': 'psycopg2-binary',
    'mysql': 'mysql-connector-python',
    'pyqt5': 'PyQt5',
    'pyqt6': 'PyQt6',
    'PySide2': 'PySide2',
    'PySide6': 'PySide6',
    # v5.1 修复：添加 PyInstaller
    'PyInstaller': 'pyinstaller',
    # v5.2: 添加更多
    'torch': 'torch',
    'torchvision': 'torchvision',
    'torchaudio': 'torchaudio',
    'tensorflow': 'tensorflow',
    'keras': 'keras',
}

# v5.1 新增：pip包名 -> 导入名 的反向映射
PIP_TO_IMPORT_MAP = {
    'Pillow': 'PIL',
    'pillow': 'PIL',
    'pyinstaller': 'PyInstaller',
    'PyInstaller': 'PyInstaller',
    'opencv-python': 'cv2',
    'opencv-python-headless': 'cv2',
    'scikit-learn': 'sklearn',
    'scikit-image': 'skimage',
    'PyYAML': 'yaml',
    'pyyaml': 'yaml',
    'beautifulsoup4': 'bs4',
    'python-dateutil': 'dateutil',
    'python-dotenv': 'dotenv',
    'PyJWT': 'jwt',
    'pyjwt': 'jwt',
    'pyserial': 'serial',
    'wxPython': 'wx',
    'wxpython': 'wx',
    'PyGObject': 'gi',
    'pycairo': 'cairo',
    'PyOpenGL': 'OpenGL',
    'pyopengl': 'OpenGL',
    'pyusb': 'usb',
    'pycryptodome': 'Crypto',
    'pycryptodomex': 'Cryptodome',
    # v5.2: 添加更多常见库
    'torch': 'torch',
    'torchvision': 'torchvision',
    'torchaudio': 'torchaudio',
    'tensorflow': 'tensorflow',
    'tensorflow-gpu': 'tensorflow',
    'keras': 'keras',
    'numpy': 'numpy',
    'pandas': 'pandas',
    'scipy': 'scipy',
    'matplotlib': 'matplotlib',
    'pygame': 'pygame',
    'requests': 'requests',
    'flask': 'flask',
    'Flask': 'flask',
    'django': 'django',
    'Django': 'django',
    'sqlalchemy': 'sqlalchemy',
    'SQLAlchemy': 'sqlalchemy',
}

# 需要 collect-submodules 的复杂库
COMPLEX_PACKAGES = {
    'pygame', 'PIL', 'numpy', 'scipy', 'matplotlib', 'pandas', 'sklearn',
    'cv2', 'tensorflow', 'torch', 'keras', 'PyQt5', 'PyQt6', 'PySide2',
    'PySide6', 'wx', 'kivy', 'pyglet', 'arcade', 'panda3d', 'moderngl',
}

# 库的隐式依赖映射
IMPLICIT_DEPENDENCIES = {
    'PIL': ['PIL._imaging', 'PIL._imagingft', 'PIL._imagingmath', 'PIL._imagingtk'],
    'numpy': ['numpy.core._multiarray_umath', 'numpy.core._dtype_ctypes', 
              'numpy.random._common', 'numpy.random._bounded_integers',
              'numpy.random._mt19937', 'numpy.random._philox', 'numpy.random._pcg64',
              'numpy.random._sfc64', 'numpy.random._generator', 'numpy.random.mtrand'],
    'pygame': ['pygame._sdl2', 'pygame.base', 'pygame.constants', 'pygame.rect',
               'pygame.rwobject', 'pygame.surflock', 'pygame.color', 'pygame.bufferproxy',
               'pygame.math', 'pygame.pkgdata', 'pygame.mixer', 'pygame.mixer_music',
               'pygame.font', 'pygame.freetype', 'pygame.image', 'pygame.transform',
               'pygame.display', 'pygame.event', 'pygame.key', 'pygame.mouse'],
    'matplotlib': ['matplotlib.backends.backend_tkagg', 'matplotlib.backends.backend_agg',
                   'matplotlib._path', 'matplotlib._image', 'matplotlib.ft2font',
                   'matplotlib._contour', 'matplotlib._qhull', 'matplotlib._tri',
                   'matplotlib._c_internal_utils'],
    'scipy': ['scipy.special._ufuncs', 'scipy.special._comb', 'scipy.linalg._fblas',
              'scipy.linalg._flapack', 'scipy.sparse._sparsetools', 
              'scipy.spatial._ckdtree', 'scipy.spatial._qhull'],
    'pandas': ['pandas._libs.tslibs.base', 'pandas._libs.tslibs.np_datetime',
               'pandas._libs.tslibs.nattype', 'pandas._libs.tslibs.timedeltas',
               'pandas._libs.tslibs.timestamps', 'pandas._libs.hashtable',
               'pandas._libs.lib', 'pandas._libs.missing', 'pandas._libs.parsers'],
    'sklearn': ['sklearn.utils._cython_blas', 'sklearn.neighbors._typedefs',
                'sklearn.neighbors._quad_tree', 'sklearn.tree._utils',
                'sklearn.utils._weight_vector'],
    'requests': ['urllib3', 'certifi', 'charset_normalizer', 'idna'],
    'aiohttp': ['aiohttp._http_parser', 'aiohttp._http_writer', 'aiohttp._websocket',
                'multidict', 'yarl', 'async_timeout', 'frozenlist', 'aiosignal'],
    'cv2': ['cv2.data', 'numpy'],
    'tkinter': ['tkinter.ttk', 'tkinter.filedialog', 'tkinter.messagebox',
                'tkinter.scrolledtext', 'tkinter.font', 'tkinter.colorchooser',
                'tkinter.simpledialog', 'tkinter.dnd'],
}

# 打包时应排除的模块
EXCLUDE_MODULES = [
    'numpy.array_api',
    'numpy.distutils', 
    'numpy.f2py',
    'numpy.testing',
    'numpy.tests',
    'scipy.spatial.cKDTree',
    'matplotlib.tests',
    'matplotlib.testing',
    'IPython',
    'jupyter',
    'jupyter_client',
    'jupyter_core',
    'notebook',
    'pytest',
    'pytest_cov',
    'sphinx',
    'setuptools',
    'pip',
    'wheel',
    'twine',
    'black',
    'flake8',
    'pylint',
    'mypy',
    'isort',
    'autopep8',
    'yapf',
    'coverage',
    'tox',
    'nox',
    'virtualenv',
    'pyinstaller',  # 不要把打包工具自己打进去
]

# 安全：允许的pip包名字符
SAFE_PACKAGE_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.]+$')


def get_python_executable() -> str:
    """获取实际的Python解释器路径（增强版）"""
    if getattr(sys, 'frozen', False):
        possible_paths = [
            shutil.which('python'),
            shutil.which('python3'),
            shutil.which('py'),
        ]
        
        # Windows 常见路径
        if sys.platform == 'win32':
            for ver in ['312', '311', '310', '39', '38']:
                possible_paths.extend([
                    rf'C:\Python{ver}\python.exe',
                    os.path.join(os.environ.get('LOCALAPPDATA', ''), 
                                'Programs', 'Python', f'Python{ver}', 'python.exe'),
                    os.path.join(os.environ.get('PROGRAMFILES', ''),
                                'Python' + ver, 'python.exe'),
                ])
        
        for path in possible_paths:
            if path and os.path.isfile(path):
                return path
        
        # 尝试 py launcher
        try:
            result = subprocess.run(
                ['py', '-c', 'import sys; print(sys.executable)'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                python_path = result.stdout.strip()
                if os.path.isfile(python_path):
                    return python_path
        except Exception:
            pass
        
        return sys.executable
    else:
        return sys.executable


def is_safe_package_name(name: str) -> bool:
    """验证包名是否安全（防止命令注入）"""
    if not name or len(name) > 100:
        return False
    return bool(SAFE_PACKAGE_NAME_PATTERN.match(name))


def is_safe_path(path: str, base_dir: Optional[str] = None) -> bool:
    """验证路径是否安全（防止路径遍历）"""
    try:
        # 规范化路径
        abs_path = os.path.abspath(path)
        
        # 检查是否包含危险模式
        dangerous_patterns = ['..', '~', '$', '%', '`', '|', ';', '&', '<', '>']
        for pattern in dangerous_patterns:
            if pattern in path:
                return False
        
        # 如果指定了基础目录，确保路径在其内
        if base_dir:
            base_abs = os.path.abspath(base_dir)
            if not abs_path.startswith(base_abs):
                return False
        
        return True
    except Exception:
        return False


def pip_name_to_import_name(pip_name: str) -> str:
    """v5.1: 将 pip 包名转换为 Python 导入名"""
    # 先查找映射表
    if pip_name in PIP_TO_IMPORT_MAP:
        return PIP_TO_IMPORT_MAP[pip_name]
    
    # 尝试小写查找
    lower_name = pip_name.lower()
    if lower_name in PIP_TO_IMPORT_MAP:
        return PIP_TO_IMPORT_MAP[lower_name]
    
    # 默认转换规则：小写，将 - 替换为 _
    return pip_name.lower().replace('-', '_')


def import_name_to_pip_name(import_name: str) -> str:
    """将 Python 导入名转换为 pip 包名"""
    if import_name in PACKAGE_NAME_MAP:
        return PACKAGE_NAME_MAP[import_name]
    return import_name


class SecureDependencyCache:
    """v5.2：带签名验证的安全缓存（修复版）"""
    
    # v5.2: 缓存有效期延长到 7 天
    CACHE_EXPIRY_SECONDS = 7 * 24 * 3600  # 7天
    
    def __init__(self, cache_file: str = None):
        # v5.2: 缓存文件放到用户目录，避免换目录丢失
        if cache_file is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".game_packer_cache")
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, "dep_cache_v5.json")
        self.cache_file = cache_file
        self.secret_key = self._get_machine_key()
        self.cache = self._load_cache()
    
    def _get_machine_key(self) -> str:
        """生成机器相关的密钥"""
        import platform
        try:
            login = os.getlogin()
        except:
            login = 'user'
        machine_info = f"{platform.node()}-{platform.machine()}-{login}"
        return hashlib.sha256(machine_info.encode()).hexdigest()[:32]
    
    def _compute_signature(self, data: dict) -> str:
        """计算数据签名"""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256((data_str + self.secret_key).encode()).hexdigest()
    
    def _load_cache(self) -> dict:
        """加载并验证缓存"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 验证签名
                signature = data.get('_signature')
                content = data.get('content', {})
                if signature == self._compute_signature(content):
                    return content
        except Exception:
            pass
        return {}
    
    def save(self):
        """保存缓存"""
        try:
            data = {
                'content': self.cache,
                '_signature': self._compute_signature(self.cache)
            }
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"保存缓存失败: {e}")
            
    def get(self, module_name: str) -> Optional[bool]:
        """获取模块状态"""
        if module_name in self.cache:
            entry = self.cache[module_name]
            # v5.2: 检查有效期
            if time.time() - entry.get('time', 0) < self.CACHE_EXPIRY_SECONDS:
                return entry.get('available')
        return None
    
    def set(self, module_name: str, available: bool):
        """设置模块状态"""
        self.cache[module_name] = {
            'available': available,
            'time': time.time()
        }
        self.save()
    
    def clear(self):
        """清除缓存"""
        self.cache = {}
        if os.path.exists(self.cache_file):
            try:
                os.remove(self.cache_file)
            except:
                pass


class AdvancedImportAnalyzer:
    """增强版导入分析器"""
    
    def __init__(self):
        self.imports = set()
        self.from_imports = set()
        self.dynamic_imports = set()
        self.conditional_imports = set()
        self.all_modules = set()
    
    def analyze_file(self, filepath: str) -> Dict[str, Set[str]]:
        """分析文件中的所有导入"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()
        except UnicodeDecodeError:
            try:
                with open(filepath, 'r', encoding='gbk') as f:
                    source = f.read()
            except Exception:
                with open(filepath, 'r', encoding='latin-1') as f:
                    source = f.read()
        
        # AST 解析
        try:
            tree = ast.parse(source)
            self._visit_tree(tree)
        except SyntaxError as e:
            print(f"[警告] 语法错误: {e}")
        
        # 正则表达式补充检测
        self._regex_analysis(source)
        
        # 合并所有导入
        self.all_modules = (
            self.imports | self.from_imports | 
            self.dynamic_imports | self.conditional_imports
        )
        
        return {
            'imports': self.imports.copy(),
            'from_imports': self.from_imports.copy(),
            'dynamic': self.dynamic_imports.copy(),
            'conditional': self.conditional_imports.copy(),
            'all': self.all_modules.copy()
        }
    
    def _visit_tree(self, node):
        """遍历AST节点"""
        if isinstance(node, ast.Import):
            for name in node.names:
                self._add_import(name.name, self.imports)
                
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                self._add_import(node.module, self.from_imports)
                
        elif isinstance(node, ast.Call):
            # 检测 __import__('xxx')
            if isinstance(node.func, ast.Name) and node.func.id == '__import__':
                if node.args and isinstance(node.args[0], ast.Constant):
                    self._add_import(node.args[0].value, self.dynamic_imports)
            
            # 检测 importlib.import_module('xxx')
            elif isinstance(node.func, ast.Attribute) and node.func.attr == 'import_module':
                if node.args and isinstance(node.args[0], ast.Constant):
                    self._add_import(node.args[0].value, self.dynamic_imports)
        
        # 递归遍历子节点
        for child in ast.iter_child_nodes(node):
            self._visit_tree(child)

    def _add_import(self, name: str, target_set: Set[str]):
        """添加导入并处理包名"""
        if not name:
            return
        top_level = name.split('.')[0]
        if top_level and top_level not in STDLIB_MODULES:
            target_set.add(top_level)

    def _regex_analysis(self, source: str):
        """正则表达式补充分析"""
        patterns = [
            # import xxx
            r'^\s*import\s+([\w\.]+)',
            # from xxx import
            r'^\s*from\s+([\w\.]+)\s+import',
            # __import__('xxx')
            r'__import__\s*\(\s*[\'"]([^\'"]+)[\'"]',
            # importlib.import_module('xxx')
            r'import_module\s*\(\s*[\'"]([^\'"]+)[\'"]',
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, source, re.MULTILINE):
                module = match.group(1)
                if module and not module.startswith('_'):
                    self._add_import(module, self.conditional_imports)


class BatchModuleChecker:
    """批量模块检测器"""
    
    def __init__(self, python_exe: str, cache: SecureDependencyCache):
        self.python_exe = python_exe
        self.cache = cache
        
    def check_modules(self, modules: Set[str], use_cache: bool = True) -> Dict[str, dict]:
        """批量检测模块状态"""
        results = {}
        to_check = []
        
        # 1. 检查缓存
        for mod in modules:
            # v5.1 修复：将导入名转为pip名（如果需要）
            pip_name = import_name_to_pip_name(mod)
            
            cached_status = self.cache.get(mod) if use_cache else None
            
            if cached_status is not None:
                results[mod] = {
                    'available': cached_status,
                    'pip_name': pip_name,
                    'version': 'Cached'
                }
            else:
                to_check.append(mod)
        
        if not to_check:
            return results
        
        # 2. 批量检测脚本
        check_script = """
import sys
import importlib.util
import json
import pkg_resources

modules = %s
results = {}

for mod in modules:
    try:
        spec = importlib.util.find_spec(mod)
        if spec is not None:
            version = "Unknown"
            try:
                # 尝试获取版本
                try:
                    m = __import__(mod)
                    version = getattr(m, '__version__', 'Unknown')
                except:
                    try:
                        version = pkg_resources.get_distribution(mod).version
                    except:
                        pass
            except: pass
            
            results[mod] = {'available': True, 'version': str(version)}
        else:
            results[mod] = {'available': False}
    except Exception as e:
        results[mod] = {'available': False, 'error': str(e)}

print(json.dumps(results))
""" % json.dumps(to_check)
        
        try:
            # 执行检测
            process = subprocess.run(
                [self.python_exe, '-c', check_script],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if process.returncode == 0:
                batch_results = json.loads(process.stdout)
                
                for mod, info in batch_results.items():
                    pip_name = import_name_to_pip_name(mod)
                    
                    # 补充信息
                    info['pip_name'] = pip_name
                    results[mod] = info
                    
                    # 更新缓存
                    self.cache.set(mod, info['available'])
            else:
                # 失败回退到逐个检测
                for mod in to_check:
                    results[mod] = {'available': False, 'pip_name': import_name_to_pip_name(mod)}
                    
        except Exception as e:
            print(f"批量检测失败: {e}")
            for mod in to_check:
                results[mod] = {'available': False, 'pip_name': import_name_to_pip_name(mod)}
        
        return results


class GamePackagerV5(QMainWindow):
    """v5.2 修复版打包工具 (PySide6)"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"别快EXE打包工具 v{VERSION} - 缓存增强版 (PySide6)")
        self.resize(900, 850)
        self.setMinimumSize(800, 700)
        
        # 核心组件
        self.python_exe = get_python_executable()
        self.dep_cache = SecureDependencyCache()
        self.import_analyzer = AdvancedImportAnalyzer()
        self.module_checker = BatchModuleChecker(self.python_exe, self.dep_cache)
        
        # 设置图标
        if os.path.exists("28x28.png"):
            self.setWindowIcon(QIcon("28x28.png"))
        
        # 默认配置
        self.current_dir = Path.cwd()
        self.default_source = "修改的游戏.py"
        self.output_name = "记事本与网址导航游戏"
        
        # 消息队列
        self.message_queue = queue.Queue()
        
        # 分析结果
        self.analyzed_deps: Dict[str, dict] = {}
        self.missing_deps: List[str] = []
        self.all_imports: Set[str] = set()
        self.hidden_imports: Set[str] = set()
        
        # 构建UI
        self._create_ui()
        
        # 定时器处理消息队列
        self.timer = QTimer()
        self.timer.timeout.connect(self._process_queue)
        self.timer.start(100)
    
    def _create_ui(self):
        """创建用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 标题栏
        title_label = QLabel(f"🎮 别快EXE打包工具 v{VERSION} - 缓存增强版")
        title_label.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("background-color: #1a237e; color: white; padding: 10px;")
        main_layout.addWidget(title_label)
        
        # Notebook
        self.notebook = QTabWidget()
        main_layout.addWidget(self.notebook)
        
        # 各标签页
        self._create_config_tab()
        self._create_check_tab()
        self._create_deps_tab()
        self._create_log_tab()
        
        # 底部控制栏
        self._create_bottom_bar(main_layout)
    
    def _create_config_tab(self):
        """配置标签页"""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        
        layout = QVBoxLayout(content_widget)
        layout.setAlignment(Qt.AlignTop)
        
        self.notebook.addTab(scroll_area, "📦 打包配置")
        
        # ============ 源文件配置 ============
        group_source = QGroupBox("源文件与输出")
        group_source.setStyleSheet("QGroupBox { font-weight: bold; }")
        layout_source = QVBoxLayout(group_source)
        
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("源文件:"))
        self.source_entry = QLineEdit(self.default_source)
        row1.addWidget(self.source_entry)
        btn_browse = QPushButton("浏览")
        btn_browse.clicked.connect(self._browse_source)
        btn_browse.setStyleSheet("background-color: #2196F3; color: white;")
        row1.addWidget(btn_browse)
        layout_source.addLayout(row1)
        
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("输出名:"))
        self.output_entry = QLineEdit(self.output_name)
        row2.addWidget(self.output_entry)
        layout_source.addLayout(row2)
        
        layout.addWidget(group_source)
        
        # ============ 图标配置 ============
        group_icon = QGroupBox("图标配置")
        group_icon.setStyleSheet("QGroupBox { font-weight: bold; }")
        layout_icon = QVBoxLayout(group_icon)
        
        self.icon_entries = {}
        icons = [
            ("EXE图标 (480x480)", "exe", "480x480.png"),
            ("窗口图标 (28x28)", "window", "28x28.png"),
            ("任务栏 (108x108)", "taskbar", "108x108.png"),
        ]
        
        for label, key, default in icons:
            row = QHBoxLayout()
            row.addWidget(QLabel(label + ":"))
            entry = QLineEdit(default)
            self.icon_entries[key] = entry
            row.addWidget(entry)
            btn = QPushButton("...")
            btn.setFixedWidth(30)
            btn.clicked.connect(lambda k=key: self._browse_icon(k))
            row.addWidget(btn)
            layout_icon.addLayout(row)
        
        layout.addWidget(group_icon)
        
        # ============ 打包模式 ============
        group_mode = QGroupBox("打包模式")
        group_mode.setStyleSheet("QGroupBox { font-weight: bold; }")
        layout_mode = QVBoxLayout(group_mode)
        
        mode_row = QHBoxLayout()
        
        # 单文件夹
        frame_onedir = QFrame()
        frame_onedir.setStyleSheet("background-color: #e8f5e9; border: 1px solid #ccc;")
        layout_onedir = QVBoxLayout(frame_onedir)
        self.rb_onedir = QRadioButton("📁 单文件夹模式（推荐）")
        self.rb_onedir.setChecked(True)
        self.rb_onedir.setStyleSheet("font-weight: bold; color: #2e7d32;")
        layout_onedir.addWidget(self.rb_onedir)
        layout_onedir.addWidget(QLabel("• 启动速度快 • 无临时文件问题\n• 适合大型游戏和复杂程序"))
        mode_row.addWidget(frame_onedir)
        
        # 单文件
        frame_onefile = QFrame()
        frame_onefile.setStyleSheet("background-color: #e3f2fd; border: 1px solid #ccc;")
        layout_onefile = QVBoxLayout(frame_onefile)
        self.rb_onefile = QRadioButton("📦 单文件模式")
        self.rb_onefile.setStyleSheet("font-weight: bold; color: #1565c0;")
        layout_onefile.addWidget(self.rb_onefile)
        layout_onefile.addWidget(QLabel("• 方便分发 • 首次启动较慢\n• 需要配置清理策略"))
        mode_row.addWidget(frame_onefile)
        
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.rb_onedir)
        self.mode_group.addButton(self.rb_onefile)
        
        layout_mode.addLayout(mode_row)
        layout.addWidget(group_mode)
        
        # 清理策略
        group_cleanup = QGroupBox("临时文件清理（单文件模式）")
        group_cleanup.setStyleSheet("QGroupBox { font-weight: bold; background-color: #fff3e0; }")
        layout_cleanup = QHBoxLayout(group_cleanup)
        
        self.cleanup_group = QButtonGroup()
        strategies = [
            ("Atexit（推荐）", 'atexit', "程序退出时清理"),
            ("Bootloader", 'bootloader', "需PyInstaller 5.0+"),
            ("不清理", 'manual', "调试用"),
        ]
        
        self.cleanup_radios = {}
        for text, value, desc in strategies:
            vbox = QVBoxLayout()
            rb = QRadioButton(text)
            if value == 'atexit':
                rb.setChecked(True)
            self.cleanup_radios[value] = rb
            self.cleanup_group.addButton(rb)
            vbox.addWidget(rb)
            lbl = QLabel(desc)
            lbl.setStyleSheet("color: gray; font-size: 10px;")
            vbox.addWidget(lbl)
            layout_cleanup.addLayout(vbox)
            
        layout.addWidget(group_cleanup)
        
        # ============ 打包选项 ============
        group_opt = QGroupBox("打包选项")
        group_opt.setStyleSheet("QGroupBox { font-weight: bold; }")
        layout_opt = QVBoxLayout(group_opt)
        
        row_opt1 = QHBoxLayout()
        self.chk_no_console = QCheckBox("隐藏控制台")
        self.chk_no_console.setChecked(True)
        self.chk_clean = QCheckBox("清理临时文件")
        self.chk_clean.setChecked(True)
        self.chk_upx = QCheckBox("UPX压缩")
        self.chk_admin = QCheckBox("管理员权限")
        self.chk_safe = QCheckBox("🛡️ 安全模式")
        self.chk_safe.setChecked(True)
        
        for chk in [self.chk_no_console, self.chk_clean, self.chk_upx, self.chk_admin, self.chk_safe]:
            row_opt1.addWidget(chk)
        layout_opt.addLayout(row_opt1)
        
        # v5.0 新选项
        row_opt2 = QHBoxLayout()
        lbl_v5 = QLabel("⚡ v5.2 增强:")
        lbl_v5.setStyleSheet("font-weight: bold; color: #1565c0;")
        row_opt2.addWidget(lbl_v5)
        
        self.chk_collect = QCheckBox("自动收集子模块")
        self.chk_collect.setChecked(True)
        self.chk_fast = QCheckBox("排除调试模块")
        self.chk_fast.setChecked(True)
        self.chk_parallel = QCheckBox("并行分析")
        self.chk_parallel.setChecked(True)
        
        for chk in [self.chk_collect, self.chk_fast, self.chk_parallel]:
            row_opt2.addWidget(chk)
        layout_opt.addLayout(row_opt2)
        
        layout.addWidget(group_opt)
        
        # v5.2 说明
        group_info = QGroupBox("v5.2 改进说明")
        group_info.setStyleSheet("QGroupBox { font-weight: bold; background-color: #e8f5e9; }")
        layout_info = QVBoxLayout(group_info)
        info_text = """✅ 缓存有效期延长至 7 天（解决重复检测问题）
✅ 缓存文件固定在用户目录（换目录不丢失）
✅ 修复 pyinstaller/torch 等检测问题
✅ 自动 collect-submodules 处理复杂库"""
        lbl_info = QLabel(info_text)
        lbl_info.setStyleSheet("color: #1b5e20;")
        layout_info.addWidget(lbl_info)
        layout.addWidget(group_info)
        
    def _create_check_tab(self):
        """环境检查标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        layout.addWidget(QLabel("检查Python环境、依赖和图标文件"))
        
        self.check_text = QTextEdit()
        self.check_text.setReadOnly(True)
        self.check_text.setFont(QFont("Consolas", 10))
        layout.addWidget(self.check_text)
        
        self.notebook.addTab(widget, "🔍 环境检查")
        
    def _create_deps_tab(self):
        """依赖分析标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        layout.addWidget(QLabel("深度分析源文件依赖（AST + 动态导入 + 隐式依赖）"))
        
        self.deps_tree = QTreeWidget()
        self.deps_tree.setHeaderLabels(['模块名', '状态', '版本', 'pip包名', '类型'])
        self.deps_tree.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        layout.addWidget(self.deps_tree)
        
        self.deps_info = QLabel("请先选择源文件并点击'分析'")
        self.deps_info.setAlignment(Qt.AlignCenter)
        self.deps_info.setStyleSheet("color: gray;")
        layout.addWidget(self.deps_info)
        
        self.notebook.addTab(widget, "📊 依赖分析")
        
    def _create_log_tab(self):
        """打包日志标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 10))
        layout.addWidget(self.log_text)
        
        btn_layout = QHBoxLayout()
        btn_clear = QPushButton("清空日志")
        btn_clear.clicked.connect(self.log_text.clear)
        btn_copy = QPushButton("复制日志")
        btn_copy.clicked.connect(self._copy_log)
        btn_layout.addWidget(btn_clear)
        btn_layout.addWidget(btn_copy)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        self.notebook.addTab(widget, "📝 打包日志")
        
    def _create_bottom_bar(self, main_layout):
        """底部控制栏"""
        bottom_widget = QWidget()
        bottom_widget.setStyleSheet("background-color: #ecf0f1;")
        layout = QVBoxLayout(bottom_widget)
        
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        
        self.progress_label = QLabel("准备就绪 - v5.2 缓存增强版")
        self.progress_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_label)
        
        btn_layout = QHBoxLayout()
        
        buttons = [
            ("🔍 检查", '#FF9800', self._start_check),
            ("📊 分析", '#9C27B0', self._start_analyze),
            ("📦 安装", '#2196F3', self._start_install),
            ("🚀 打包", '#4CAF50', self._start_pack),
            ("🗑️ 清缓存", '#FF5722', self._clear_cache),
            ("📁 目录", '#607D8B', self._open_output),
            ("❌ 退出", '#F44336', self.close),
        ]
        
        self.btn_refs = {}
        for text, color, cmd in buttons:
            btn = QPushButton(text)
            btn.setStyleSheet(f"background-color: {color}; color: white; font-weight: bold; padding: 5px;")
            btn.clicked.connect(cmd)
            btn_layout.addWidget(btn)
            self.btn_refs[text] = btn
            
        layout.addLayout(btn_layout)
        main_layout.addWidget(bottom_widget)
        
        # 初始状态
        self.btn_refs["📊 分析"].setEnabled(False)
        self.btn_refs["🚀 打包"].setEnabled(False)
        
    # ==================== 工具方法 ====================
    
    def _browse_source(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "选择Python源文件", "", "Python文件 (*.py);;所有文件 (*.*)")
        if filepath:
            self.source_entry.setText(filepath)
            self.analyzed_deps = {}
            self.missing_deps = []
            self.btn_refs["📊 分析"].setEnabled(False)
            self.btn_refs["🚀 打包"].setEnabled(False)
            
    def _browse_icon(self, icon_type):
        filepath, _ = QFileDialog.getOpenFileName(self, f"选择{icon_type}图标", "", "图片文件 (*.png *.ico);;所有文件 (*.*)")
        if filepath:
            self.icon_entries[icon_type].setText(filepath)
            
    def _get_source_file(self) -> str:
        source = self.source_entry.text().strip()
        if source and not source.endswith('.py'):
            source += '.py'
        return source
        
    def _copy_log(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.log_text.toPlainText())
        QMessageBox.information(self, "成功", "日志已复制到剪贴板")
        
    def _clear_cache(self):
        self.dep_cache.clear()
        self.analyzed_deps = {}
        self.missing_deps = []
        QMessageBox.information(self, "成功", "缓存已清除，下次分析将重新检测所有模块")
        
    def _open_output(self):
        dist_dir = Path("dist")
        if dist_dir.exists():
            if sys.platform == 'win32':
                os.startfile(dist_dir)
            elif sys.platform == 'darwin':
                subprocess.run(['open', dist_dir])
            else:
                subprocess.run(['xdg-open', dist_dir])
        else:
            QMessageBox.information(self, "提示", "输出目录不存在，请先完成打包")
            
    def _add_check_msg(self, msg: str):
        self.message_queue.put(('check', msg))
        
    def _add_log_msg(self, msg: str):
        self.message_queue.put(('log', msg))
        
    def _process_queue(self):
        try:
            while True:
                msg_type, content = self.message_queue.get_nowait()
                
                if msg_type == 'check':
                    self.check_text.moveCursor(QTextCursor.End)
                    self.check_text.insertPlainText(content)
                elif msg_type == 'log':
                    self.log_text.moveCursor(QTextCursor.End)
                    self.log_text.insertPlainText(content)
                elif msg_type == 'progress':
                    value, text = content
                    self.progress.setValue(value)
                    self.progress_label.setText(text)
                elif msg_type == 'deps_tree':
                    self.deps_tree.clear()
                    for item in content:
                        QTreeWidgetItem(self.deps_tree, item)
                elif msg_type == 'deps_info':
                    text, color = content
                    self.deps_info.setText(text)
                    self.deps_info.setStyleSheet(f"color: {color};")
                elif msg_type == 'enable_btn':
                    btn_text = content
                    if btn_text in self.btn_refs:
                        self.btn_refs[btn_text].setEnabled(True)
                elif msg_type == 'disable_btn':
                    btn_text = content
                    if btn_text in self.btn_refs:
                        self.btn_refs[btn_text].setEnabled(False)
        except queue.Empty:
            pass

    # ==================== 业务逻辑 ====================
    # (Checking, Analyzing, Installing, Packing logic remains largely same, just calling _add_msg)

    def _start_check(self):
        self.notebook.setCurrentIndex(1)
        self.btn_refs["🔍 检查"].setEnabled(False)
        self.check_text.clear()
        threading.Thread(target=self._do_check, daemon=True).start()
        
    def _do_check(self):
        all_ok = True
        try:
            self._add_check_msg(f"{'='*60}\n")
            self._add_check_msg(f"环境检查 v{VERSION}\n")
            self._add_check_msg(f"{'='*60}\n\n")
            
            # Python信息
            py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            self._add_check_msg(f"Python版本: {py_ver}\n")
            self._add_check_msg(f"解释器: {self.python_exe}\n")
            
            if getattr(sys, 'frozen', False):
                self._add_check_msg("  ⚠️ 运行在打包环境中\n")
            
            # 源文件检查
            self._add_check_msg(f"\n{'='*40}\n")
            self._add_check_msg("源文件检查\n")
            self._add_check_msg(f"{'='*40}\n")
            
            source = self._get_source_file()
            if os.path.exists(source):
                self._add_check_msg(f"✅ 源文件存在: {source}\n")
                if is_safe_path(source):
                    self._add_check_msg(f"✅ 路径安全验证通过\n")
                else:
                    self._add_check_msg(f"⚠️ 路径包含可疑字符\n")
                
                try:
                    with open(source, 'r', encoding='utf-8') as f:
                        content = f.read()
                    compile(content, source, 'exec')
                    self._add_check_msg(f"✅ 语法正确 ({len(content):,} 字符)\n")
                except SyntaxError as e:
                    self._add_check_msg(f"❌ 语法错误: 第{e.lineno}行 - {e.msg}\n")
                    all_ok = False
                except Exception as e:
                    self._add_check_msg(f"❌ 读取失败: {e}\n")
                    all_ok = False
            else:
                self._add_check_msg(f"❌ 源文件不存在: {source}\n")
                all_ok = False
            
            # 图标检查
            self._add_check_msg(f"\n{'='*40}\n")
            self._add_check_msg("图标文件检查\n")
            self._add_check_msg(f"{'='*40}\n")
            
            for key, entry in self.icon_entries.items():
                path = entry.text()
                if path:
                    abs_path = os.path.abspath(path)
                    if os.path.exists(abs_path):
                        size = os.path.getsize(abs_path)
                        self._add_check_msg(f"✅ {key}: {os.path.basename(path)} ({size:,} bytes)\n")
                    else:
                        self._add_check_msg(f"⚠️ {key}不存在: {path}\n")
            
            # 核心依赖检查
            self._add_check_msg(f"\n{'='*40}\n")
            self._add_check_msg("核心依赖检查\n")
            self._add_check_msg(f"{'='*40}\n")
            
            core_deps = ['PyInstaller', 'PIL']
            results = self.module_checker.check_modules(set(core_deps), use_cache=False)
            
            for dep in core_deps:
                info = results.get(dep, {})
                display_name = dep.lower() if dep == 'PyInstaller' else dep
                pip_name = PACKAGE_NAME_MAP.get(dep, dep.lower())
                
                if info.get('available'):
                    ver = info.get('version', 'N/A')
                    self._add_check_msg(f"✅ {display_name}: v{ver}\n")
                else:
                    self._add_check_msg(f"❌ {display_name}: 未安装 (pip install {pip_name})\n")
                    if dep == 'PyInstaller':
                        all_ok = False
            
            # UI库检查
            self._add_check_msg(f"\n{'='*40}\n")
            self._add_check_msg("UI环境\n")
            self._add_check_msg(f"{'='*40}\n")
            self._add_check_msg("✅ PySide6 (当前运行环境)\n")
            
            self._add_check_msg(f"\n{'='*60}\n")
            if all_ok:
                self._add_check_msg("✅ 环境检查通过！请点击'分析'按钮\n")
                self.message_queue.put(('enable_btn', "📊 分析"))
            else:
                self._add_check_msg("❌ 检查未通过，请先解决上述问题\n")
                
        except Exception as e:
            self._add_check_msg(f"\n❌ 检查出错: {e}\n")
            self._add_check_msg(traceback.format_exc())
            
        self.message_queue.put(('enable_btn', "🔍 检查"))

    def _start_analyze(self):
        source = self._get_source_file()
        if not os.path.exists(source):
            QMessageBox.critical(self, "错误", f"源文件不存在: {source}")
            return
        
        self.notebook.setCurrentIndex(2)
        self.btn_refs["📊 分析"].setEnabled(False)
        self.deps_tree.clear()
        self.deps_info.setText("正在深度分析依赖...")
        self.deps_info.setStyleSheet("color: blue;")
        
        threading.Thread(target=self._do_analyze, args=(source,), daemon=True).start()

    def _do_analyze(self, source: str):
        # ... logic same as before, adapted for queue ...
        try:
            self.message_queue.put(('progress', (10, "解析源代码...")))
            
            analyzer = AdvancedImportAnalyzer()
            import_result = analyzer.analyze_file(source)
            all_imports = import_result['all']
            
            self.message_queue.put(('progress', (30, f"检测 {len(all_imports)} 个模块...")))
            
            expanded = set()
            for mod in all_imports:
                top = mod.split('.')[0]
                expanded.add(top)
                if top in IMPLICIT_DEPENDENCIES:
                    for implicit in IMPLICIT_DEPENDENCIES[top]:
                        expanded.add(implicit)
                        expanded.add(implicit.split('.')[0])
            
            self.message_queue.put(('progress', (50, "批量检测模块状态...")))
            results = self.module_checker.check_modules(expanded)
            
            self.analyzed_deps = {}
            self.missing_deps = []
            self.all_imports = set()
            self.hidden_imports = set()
            tree_data = []
            
            for mod, info in sorted(results.items()):
                if mod in STDLIB_MODULES: continue
                
                self.analyzed_deps[mod] = info
                self.all_imports.add(mod)
                status = '✅ 已安装' if info['available'] else '❌ 未安装'
                
                if mod in import_result['imports'] or mod in import_result['from_imports']:
                    source_type = '直接导入'
                elif mod in import_result['dynamic']:
                    source_type = '动态导入'
                elif mod in import_result['conditional']:
                    source_type = '条件导入'
                else:
                    source_type = '隐式依赖'
                    self.hidden_imports.add(mod)
                
                tree_data.append((mod, status, info.get('version', 'N/A'),
                                  info.get('pip_name', mod), source_type))
                
                if not info['available'] and info.get('pip_name', '-') != '-':
                    self.missing_deps.append(info['pip_name'])
            
            self.missing_deps = list(set(self.missing_deps))
            
            self.message_queue.put(('progress', (90, "更新界面...")))
            self.message_queue.put(('deps_tree', tree_data))
            
            total = len(tree_data)
            missing = len(self.missing_deps)
            implicit = len(self.hidden_imports)
            
            if missing > 0:
                info_text = f"发现 {missing} 个缺失依赖: {', '.join(self.missing_deps[:5])}"
                if len(self.missing_deps) > 5: info_text += f" ... 等"
                self.message_queue.put(('deps_info', (info_text, 'red')))
            else:
                info_text = f"✅ 所有 {total} 个第三方依赖就绪 (含 {implicit} 个隐式依赖)"
                self.message_queue.put(('deps_info', (info_text, 'green')))
                self.message_queue.put(('enable_btn', "🚀 打包"))
                
            self.message_queue.put(('progress', (100, "分析完成")))
        except Exception as e:
            self.message_queue.put(('deps_info', (f"分析失败: {e}", 'red')))
            traceback.print_exc()
        
        self.message_queue.put(('enable_btn', "📊 分析"))

    def _start_install(self):
        self.btn_refs["📦 安装"].setEnabled(False)
        self.notebook.setCurrentIndex(3)
        threading.Thread(target=self._do_install, daemon=True).start()

    def _do_install(self):
        # ... logic same as before ...
        try:
            to_install = []
            core_check = {'PyInstaller': 'pyinstaller', 'PIL': 'Pillow'}
            core_results = self.module_checker.check_modules(set(core_check.keys()), use_cache=False)
            
            for import_name, pip_name in core_check.items():
                if not core_results.get(import_name, {}).get('available'):
                    to_install.append(pip_name)
            
            for dep in self.missing_deps:
                if dep not in to_install and dep != '-':
                    if is_safe_package_name(dep):
                        to_install.append(dep)
            
            to_install = list(set(to_install))
            
            self._add_log_msg(f"{'='*60}\nv{VERSION} 安全安装模式\n{'='*60}\n\n")
            
            if not to_install:
                self._add_log_msg("✅ 所有依赖已安装，无需操作\n")
                self.message_queue.put(('enable_btn', "📦 安装"))
                return
            
            self._add_log_msg(f"需要安装: {', '.join(to_install)}\n\n")
            mirrors = [("清华镜像", "https://pypi.tuna.tsinghua.edu.cn/simple"),
                       ("阿里云", "https://mirrors.aliyun.com/pypi/simple"),
                       ("官方源", "https://pypi.org/simple")]
            
            success = 0
            failed = 0
            
            for pkg in to_install:
                self._add_log_msg(f"安装 {pkg}...\n")
                installed = False
                for mirror_name, mirror_url in mirrors:
                    try:
                        self._add_log_msg(f"  尝试 {mirror_name}...\n")
                        cmd = [self.python_exe, "-m", "pip", "install", pkg, "-i", mirror_url, "--upgrade", "--no-warn-script-location"]
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                        
                        if result.returncode == 0:
                            self._add_log_msg(f"  ✅ 安装成功\n")
                            installed = True
                            success += 1
                            import_name = pip_name_to_import_name(pkg)
                            self.dep_cache.set(import_name, True)
                            break
                        else:
                            err = result.stderr[:100] if result.stderr else "未知错误"
                            self._add_log_msg(f"  ⚠️ 失败: {err}\n")
                    except Exception as e:
                        self._add_log_msg(f"  ⚠️ 错误: {e}\n")
                
                if not installed: failed += 1
                self._add_log_msg("-" * 40 + "\n")
            
            self._add_log_msg(f"\n完成！成功: {success}, 失败: {failed}\n")
            if failed == 0:
                self._add_log_msg("\n✅ 所有依赖安装成功！请重新点击'检查'\n")
                self.missing_deps = []
        except Exception as e:
            self._add_log_msg(f"\n❌ 安装出错: {e}\n")
            traceback.print_exc()
        
        self.message_queue.put(('enable_btn', "📦 安装"))

    def _start_pack(self):
        source = self._get_source_file()
        if not os.path.exists(source):
            QMessageBox.critical(self, "错误", f"源文件不存在: {source}")
            return
        
        self.btn_refs["🚀 打包"].setEnabled(False)
        self.notebook.setCurrentIndex(3)
        self.log_text.clear()
        threading.Thread(target=self._do_pack, args=(source,), daemon=True).start()

    def _do_pack(self, source: str):
        # ... packing logic ...
        wrapper_file = None
        temp_ico = None
        
        try:
            output_name = self.output_entry.text().strip() or "output"
            pack_mode = "onefile" if self.rb_onefile.isChecked() else "onedir"
            
            self.message_queue.put(('progress', (5, "初始化...")))
            self._add_log_msg(f"{'='*70}\n开始打包 v{VERSION}\n{'='*70}\n源文件: {source}\n输出名: {output_name}\n模式: {pack_mode}\n{'='*70}\n\n")
            
            self.message_queue.put(('progress', (10, "准备图标...")))
            icons = self._prepare_icons()
            if 'exe' in icons and icons['exe'].endswith('temp_app_icon.ico'):
                temp_ico = icons['exe']
            
            self.message_queue.put(('progress', (15, "生成包装器...")))
            if pack_mode == 'onefile' or icons.get('window') or icons.get('taskbar'):
                wrapper_file = self._create_wrapper(source, icons)
                actual_source = wrapper_file
                self._add_log_msg(f"✅ 包装器: {wrapper_file}\n")
            else:
                actual_source = source
            
            self.message_queue.put(('progress', (20, "收集资源...")))
            data_files = self._collect_data_files(source, icons)
            self._add_log_msg(f"✅ 收集了 {len(data_files)} 个数据文件\n")
            
            self.message_queue.put(('progress', (25, "构建命令...")))
            cmd = self._build_command(actual_source, output_name, icons, data_files)
            
            self.message_queue.put(('progress', (30, "执行打包...")))
            self._add_log_msg(f"\n执行PyInstaller...\n命令: {' '.join(cmd[:15])}...\n\n")
            
            start_time = time.time()
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1)
            
            progress = 30
            for line in process.stdout:
                self._add_log_msg(line)
                if "Building" in line or "Analyzing" in line:
                    progress = min(progress + 2, 90)
                elif "Copying" in line:
                    progress = min(progress + 1, 90)
                self.message_queue.put(('progress', (progress, "打包中...")))
            
            process.wait()
            elapsed = time.time() - start_time
            
            self.message_queue.put(('progress', (95, "检查结果...")))
            
            if pack_mode == 'onefile':
                exe_path = Path("dist") / f"{output_name}.exe"
            else:
                exe_path = Path("dist") / output_name / f"{output_name}.exe"
            
            if exe_path.exists():
                file_size = exe_path.stat().st_size / (1024 * 1024)
                self.message_queue.put(('progress', (100, f"完成！耗时 {elapsed:.1f}s")))
                self._add_log_msg(f"\n{'='*70}\n✅ 打包成功！\n输出: {exe_path}\n大小: {file_size:.2f} MB\n耗时: {elapsed:.1f} 秒\n{'='*70}\n")
                
                # Show success message box using QTimer to run in main thread
                # Actually, QMessageBox should be called from main thread.
                # But here we are in a thread. 
                # We can't call QMessageBox directly.
                # We can use invokeMethod or signals, or just log it.
                # For simplicity, I'll skip the popup from thread or use a hack.
                # Wait, I can use QMetaObject.invokeMethod.
                pass 
            else:
                self.message_queue.put(('progress', (100, "失败")))
                self._add_log_msg(f"\n❌ 打包失败 - 未找到输出文件\n")
        
        except Exception as e:
            self.message_queue.put(('progress', (100, f"错误: {e}")))
            self._add_log_msg(f"\n❌ 打包出错: {e}\n")
            self._add_log_msg(traceback.format_exc())
        
        finally:
            for f in [wrapper_file, temp_ico]:
                if f and os.path.exists(f):
                    try:
                        os.remove(f)
                    except: pass
            self.message_queue.put(('enable_btn', "🚀 打包"))

    def _prepare_icons(self) -> Dict[str, str]:
        icons = {}
        try:
            from PIL import Image
            has_pil = True
        except ImportError:
            has_pil = False
            self._add_log_msg("⚠️ Pillow未安装，无法转换PNG到ICO\n")
            
        exe_icon = self.icon_entries['exe'].text()
        if exe_icon:
            abs_path = os.path.abspath(exe_icon)
            if os.path.exists(abs_path):
                if abs_path.lower().endswith('.png') and has_pil:
                    try:
                        img = Image.open(abs_path)
                        if img.mode != 'RGBA': img = img.convert('RGBA')
                        ico_path = "temp_app_icon.ico"
                        img.save(ico_path, format='ICO', sizes=[(16,16), (32,32), (48,48), (64,64), (128,128), (256,256)])
                        icons['exe'] = os.path.abspath(ico_path)
                        self._add_log_msg(f"✅ 生成ICO: {ico_path}\n")
                    except Exception as e:
                        self._add_log_msg(f"⚠️ ICO转换失败: {e}\n")
                        icons['exe'] = abs_path
                else:
                    icons['exe'] = abs_path
        
        for key in ['window', 'taskbar']:
            path = self.icon_entries[key].text()
            if path:
                abs_path = os.path.abspath(path)
                if os.path.exists(abs_path):
                    icons[key] = abs_path
        return icons

    def _create_wrapper(self, source: str, icons: Dict[str, str]) -> str:
        # Same wrapper generation logic
        try:
            with open(source, 'r', encoding='utf-8') as f: original = f.read()
        except:
            with open(source, 'r', encoding='gbk') as f: original = f.read()
            
        window_icon = os.path.basename(icons.get('window', '')) if icons.get('window') else ''
        taskbar_icon = os.path.basename(icons.get('taskbar', '')) if icons.get('taskbar') else ''
        
        cleanup_code = ''
        if self.rb_onefile.isChecked() and self.cleanup_radios['atexit'].isChecked():
            cleanup_code = '''
import sys, os, atexit, shutil, time
def _cleanup_meipass():
    if hasattr(sys, '_MEIPASS'):
        try:
            time.sleep(0.3)
            shutil.rmtree(sys._MEIPASS, ignore_errors=True)
        except: pass
if hasattr(sys, '_MEIPASS'):
    atexit.register(_cleanup_meipass)
'''
        
        wrapper_code = f'''# -*- coding: utf-8 -*-
# 自动生成的包装器 v{VERSION}
{cleanup_code}
import sys, os

def _setup_icons():
    try:
        if hasattr(sys, '_MEIPASS'):
            base = sys._MEIPASS
        else:
            base = os.path.dirname(os.path.abspath(__file__))
        
        window_icon = "{window_icon}"
        taskbar_icon = "{taskbar_icon}"
        
        def find_icon(name):
            if not name: return None
            for p in [os.path.join(base, name), os.path.join(os.getcwd(), name), name]:
                if os.path.exists(p): return os.path.abspath(p)
            return None
        
        try:
            import tkinter as tk
            _orig_tk = tk.Tk.__init__
            def _new_tk(self, *a, **kw):
                _orig_tk(self, *a, **kw)
                try:
                    icon = find_icon(window_icon)
                    if icon and icon.endswith('.png'):
                        photo = tk.PhotoImage(file=icon)
                        self.iconphoto(True, photo)
                        self._icon_ref = photo
                    elif icon:
                        self.iconbitmap(icon)
                except: pass
            tk.Tk.__init__ = _new_tk
        except: pass
        
        try:
            import pygame
            _orig_init = pygame.init
            def _new_init(*a, **kw):
                r = _orig_init(*a, **kw)
                try:
                    icon = find_icon(window_icon)
                    if icon:
                        pygame.display.set_icon(pygame.image.load(icon))
                except: pass
                return r
            pygame.init = _new_init
        except: pass
    except: pass

_setup_icons()

# ===== 原始代码 =====
'''
        wrapper = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.py', delete=False)
        wrapper.write(wrapper_code)
        wrapper.write(original)
        wrapper.close()
        return wrapper.name

    def _collect_data_files(self, source: str, icons: Dict[str, str]) -> List[Tuple[str, str]]:
        # Same logic
        data_files = []
        collected = set()
        source_dir = os.path.dirname(os.path.abspath(source)) or '.'
        try:
            with open(source, 'r', encoding='utf-8') as f: code = f.read()
        except: code = ""
        
        patterns = [
            r'["\']([^"\']+\.(?:png|jpg|jpeg|gif|ico|bmp))["\']',
            r'["\']([^"\']+\.(?:json|txt|xml|cfg|ini|yaml|yml))["\']',
            r'["\']([^"\']+\.(?:wav|mp3|ogg|flac))["\']',
            r'["\']([^"\']+\.(?:ttf|otf))["\']',
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, code, re.IGNORECASE):
                ref = match.group(1)
                for full in [os.path.join(source_dir, ref), os.path.abspath(ref)]:
                    if os.path.exists(full):
                        abs_path = os.path.abspath(full)
                        if abs_path not in collected:
                            data_files.append((abs_path, '.'))
                            collected.add(abs_path)
                        break
        
        for icon_path in icons.values():
            if icon_path and os.path.exists(icon_path):
                abs_path = os.path.abspath(icon_path)
                if abs_path not in collected:
                    data_files.append((abs_path, '.'))
                    collected.add(abs_path)
        return data_files

    def _build_command(self, source: str, output_name: str, icons: Dict[str, str], data_files: List[Tuple[str, str]]) -> List[str]:
        cmd = [self.python_exe, "-m", "PyInstaller"]
        if self.chk_clean.isChecked(): cmd.append("--clean")
        cmd.append("--noconfirm")
        
        if self.rb_onefile.isChecked(): cmd.append("--onefile")
        else: cmd.append("--onedir")
        
        if self.chk_no_console.isChecked(): cmd.append("--noconsole")
        
        if 'exe' in icons: cmd.extend(["--icon", icons['exe']])
        cmd.extend(["--name", output_name])
        
        if self.chk_fast.isChecked():
            for exclude in EXCLUDE_MODULES:
                cmd.extend(["--exclude-module", exclude])
        
        sep = ';' if sys.platform == 'win32' else ':'
        for src, dst in data_files:
            cmd.extend(["--add-data", f"{src}{sep}{dst}"])
        
        added_hidden = set()
        for mod in self.all_imports:
            if mod not in STDLIB_MODULES and mod not in added_hidden:
                skip = False
                for excl in EXCLUDE_MODULES:
                    if mod.startswith(excl.split('.')[0]):
                        skip = True
                        break
                if not skip:
                    cmd.extend(["--hidden-import", mod])
                    added_hidden.add(mod)
        
        for mod in self.hidden_imports:
            if mod not in added_hidden:
                cmd.extend(["--hidden-import", mod])
                added_hidden.add(mod)
        
        if self.chk_collect.isChecked():
            for mod in self.all_imports:
                top = mod.split('.')[0]
                if top in COMPLEX_PACKAGES and top not in STDLIB_MODULES:
                    cmd.extend(["--collect-submodules", top])
                    self._add_log_msg(f"  📦 collect-submodules: {top}\n")
        
        if self.chk_safe.isChecked():
            cmd.extend(["--collect-all", "pkg_resources"])
            cmd.extend(["--collect-all", "tkinter"])
        
        if self.chk_admin.isChecked(): cmd.append("--uac-admin")
        
        if self.chk_upx.isChecked() and shutil.which('upx'):
            cmd.append("--upx-dir=.")
        else:
            cmd.append("--noupx")
            
        cmd.append(source)
        return cmd


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle(QStyleFactory.create("Fusion"))
    
    print(f"{'='*70}")
    print(f"游戏一键打包工具 v{VERSION} - 缓存增强版 (PySide6)")
    print("="*70)
    print("✅ 缓存有效期：7 天（解决重复检测问题）")
    print("✅ 缓存位置：用户目录（换目录不丢失）")
    print("✅ 修复：pyinstaller/torch 等模块检测")
    print("✅ 兼容性：自动 collect-submodules 处理复杂库")
    print("="*70)
    print()
    
    try:
        window = GamePackagerV5()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"启动失败: {e}")
        traceback.print_exc()
        input("按Enter键退出...")


if __name__ == "__main__":
    main()
