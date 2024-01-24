#!/usr/bin/env python3
# Copyright 2019 Mycroft AI Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from argparse import ArgumentParser
from precise.util import activate_notify
from precise.runner import PreciseRunner, PreciseEngine
from threading import Event, Thread
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as line
import matplotlib.animation as animation
import numpy as np
from collections import deque
import time


def main():
    engine = "/home/anna/WorkSpace/celadon/demo-src/precise/precise/engine.py"
    model = "/home/anna/WorkSpace/celadon/demo-src/precise/training/hello_intel.net"
    x = np.arange(0, 10, 0.1)
    y1 = deque(maxlen=100)
    y2 = deque(maxlen=100)
    y3 = deque(maxlen=100)
    for i in range(100):
        y1.append(0)
        y2.append(0.1)
        y3.append(0.5)

    fig = plt.figure()
    ax = plt.subplot(111, ylim=(0, 5))
    line1 = line.Line2D([], [], color="blue", label="raw prob")
    line2 = line.Line2D([], [], color="red", label="smooth prob")
    line3 = line.Line2D([], [], color="green", label="activate threshold")
    # leg = fig.legend(loc='upper center')
    # leg = plt.legend(loc='upper center')
    ax.set_autoscaley_on(True)
    ax.grid()

    def init_plot():
        ax.add_line(line1)
        ax.add_line(line2)
        ax.add_line(line3)

        return line1, line2, line3,

    def update_plot(i):
        line1.set_data(x, y1.copy())
        line2.set_data(x, y2.copy())
        line3.set_data(x, y3.copy())
        ax.relim()
        ax.autoscale_view()

        fig.canvas.draw()

        # fig.canvas.flush_events()
        return line1, line2, line3

    def on_prediction(raw_prob, smooth_prob):
        print("raw_prob = {:.3f}, smooth_prob = {:.3f} {}".format(raw_prob, smooth_prob, ("get" if raw_prob > 0.5 else "")))
        y1.append(raw_prob)
        y2.append(smooth_prob)

    def on_activation():
        activate_notify()

    ani = animation.FuncAnimation(fig,  # 画布
                                  update_plot,  # 图像更新
                                  init_func=init_plot,  # 图像初始化
                                  frames=24,
                                  interval=10,  # 图像更新间隔
                                  blit=True)


    engine = PreciseEngine(engine, model, chunk_size=2048)
    PreciseRunner(engine, on_prediction=on_prediction, on_activation=on_activation, trigger_level=3, sensitivity=0.5).start()
    plt.show()  # 显示图像
    Event().wait()  # Wait forever


if __name__ == '__main__':
    main()
