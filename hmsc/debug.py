import tensorflow as tf

def print_memory_consumption(label, *, reset=False):
    info = tf.config.experimental.get_memory_info("GPU:0")
    tf.print(f"{label:8s} - current:", info['current'] / 1024**3, "GiB, peak:", info['peak'] / 1024**3, "GiB")
    if reset:
        tf.config.experimental.reset_memory_stats("GPU:0")
