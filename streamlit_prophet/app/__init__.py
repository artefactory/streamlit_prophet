import os
import sys

from streamlit.web import cli


def deploy_streamlit():
    sys.argv = [
        "streamlit",
        "run",
        f"{os.path.dirname(os.path.realpath(__file__))}/dashboard.py",
        "--server.port=8080",
        "--server.address=0.0.0.0",
    ]
    sys.exit(cli.main())

def deploy_streamlit_with_base_path():
    sys.argv = [
        "streamlit",
        "run",
        f"{os.path.dirname(os.path.realpath(__file__))}/dashboard.py",
        "--server.port=8080",
        "--server.address=0.0.0.0",
        "--server.baseUrlPath=/streamlit-prophet",
    ]
    sys.exit(cli.main())
