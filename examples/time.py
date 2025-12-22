import time

start_time = time.perf_counter()

# Code to be timed goes here (e.g., a function call, a loop)
for i in range(1000000):
    pass

end_time = time.perf_counter()
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time:.4f} seconds")
