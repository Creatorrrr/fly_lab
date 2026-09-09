"""TEST ONLY. Synthetic body fixture; never called by production launchers."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aiohttp import web
from flylab.server import Server
from tests.fixture_body import FixtureBody
if __name__=='__main__':
    print('TEST DOUBLE SERVER — NO MUJOCO PHYSICS',flush=True)
    web.run_app(Server(8766,FixtureBody).app(),host='127.0.0.1',port=8766,access_log=None)
