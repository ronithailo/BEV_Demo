import time
class fps_calc:
    def __init__(self, freq=60):
        self.start_time_stamp = 0
        self.last_time_stamp = 0
        self.frame_count = -1
        self.freq = freq

    def update_fps(self):
        self.frame_count = self.frame_count + 1
        now = time.time()
        
        if self.start_time_stamp == 0:
            self.start_time_stamp = time.time()
            self.last_time_stamp = self.start_time_stamp
        if now >= self.last_time_stamp + self.freq:
            with open('FPS.log', 'a') as file:
                file.write(f'The AVG FPS is {self.frame_count / (now - self.start_time_stamp)}')
            print(f'The AVG FPS is {self.frame_count / (now - self.start_time_stamp)}')
            self.last_time_stamp = now