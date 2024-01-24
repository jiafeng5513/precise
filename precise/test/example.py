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
from threading import Event


def main():
    # parser = ArgumentParser('Implementation demo of precise-engine')
    engine = "/home/anna/WorkSpace/celadon/demo-src/precise/precise/engine.py"
    model = "/home/anna/WorkSpace/celadon/demo-src/precise/training/hello_intel.net"

    # args = parser.parse_args()

    def on_prediction(prob):
        print("prob = {:.2f} {}".format(prob, ("get" if prob > 0.5 else "")))
        # print('!' if prob > 0.5 else '.', end='', flush=True)

    def on_activation():
        activate_notify()

    engine = PreciseEngine(engine, model, chunk_size=2048)
    PreciseRunner(engine, on_prediction=on_prediction, on_activation=on_activation, trigger_level=3, sensitivity=0.5).start()
    Event().wait()  # Wait forever


if __name__ == '__main__':
    main()
