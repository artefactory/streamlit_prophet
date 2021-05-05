import os
import sys

from streamlit import cli


def deploy_streamlit():
    sys.argv = ["streamlit", "run", f"{os.path.dirname(os.path.realpath(__file__))}/dashboard.py"]
    sys.exit(cli.main())
