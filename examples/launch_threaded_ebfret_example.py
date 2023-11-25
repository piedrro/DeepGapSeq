
import time
from DeepGapSeq.GUI.ebfret_utils import ebFRET_controller

controller = ebFRET_controller()

controller.start_ebfret()
print("Closing ebFRET GUI in 5 seconds...")
time.sleep(5)

controller.close_ebfret()
