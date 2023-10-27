
import threading
import time
from DeepGapSeq.ebFRET.ebfret_utils import ebFRET_controller

controller = ebFRET_controller()

ebfret_thread = threading.Thread(target=controller.start_ebfret)
ebfret_thread.start()

while not controller.ebfret_running:
    time.sleep(0.1)

print("ebFRET GUI is running")
print("Closing ebFRET GUI in 5 seconds...")
time.sleep(5)
controller.close_ebfret()
