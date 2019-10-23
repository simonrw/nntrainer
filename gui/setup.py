from setuptools import setup, find_packages

setup(
    name="nntrainer_gui",
    author="Simon Walker",
    author_email="s.r.walker101@googlemail.com",
    version="0.0.1",
    packages=find_packages(),
    entry_points={"gui_scripts": ["nntrainer = nntrainer.app:main"]},
    install_requires=[
        "tensorflow>=2",
        "numpy",
        "pyqt5",
        "pyqtgraph",
        ],
)
