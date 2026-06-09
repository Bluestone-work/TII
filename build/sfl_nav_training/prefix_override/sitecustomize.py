import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/wj/work/multi-robot-exploration-rl/install/sfl_nav_training'
