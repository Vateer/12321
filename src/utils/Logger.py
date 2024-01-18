import os
from datetime import datetime
import atexit

class Logger():
    def __init__(self, args, make_file=True) -> None:
        self.args = args
        self.log_buffer = ""
        self.log_buffer_size_threshold = 1024
        self.path = os.path.join(args.logdir, args.dataset)
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")[2:]
        self.path = os.path.join(self.path, timestamp)
        self.path = os.path.join(self.path, "rc0")
        if make_file == True:
            if not os.path.exists(self.path) and args.log == True:
                os.makedirs(self.path, exist_ok=True)
        atexit.register(self._exit_handler)
        self.log_name = "log.txt"

    def new_cluster(self, num):
        self._write_buffer_to_file()
        self.path = self.path[:-1]+str(num)
        if not os.path.exists(self.path) and self.args.log == True:
            os.makedirs(self.path, exist_ok=True)

    def save(self, content, end="\n"):
        content = str(content)
        print(content, end=end)
        content += end
        if self.args.log == False:
            return
        self.log_buffer += content
        if len(self.log_buffer) >= self.log_buffer_size_threshold:
            self._write_buffer_to_file()

    def _write_buffer_to_file(self):
        if self.log_buffer:
            with open(os.path.join(self.path, self.log_name), "a") as file:
                file.write(self.log_buffer)
            self.log_buffer = ""

    def _exit_handler(self):
        self._write_buffer_to_file()

    def get_log_dir(self):
        return self.path
    
class SubLogger(Logger):
    def __init__(self, args, path, name) -> None:
        super().__init__(args, make_file=False)
        self.path = path
        self.log_name = name
    
    # def new_cluster(self, num):
    #     super().new_cluster(num)
    
    def save(self, content):
        content = str(content)
        content += "\n"
        if self.args.log == False:
            return
        self.log_buffer += content
        if len(self.log_buffer) >= self.log_buffer_size_threshold:
            self._write_buffer_to_file()
    def end(self):
        self._write_buffer_to_file()

