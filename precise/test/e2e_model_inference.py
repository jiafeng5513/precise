from precise.network_runner import KerasRunner

model_name = "/home/anna/WorkSpace/celadon/demo-src/precise/training/hello_intel.net"


def test_model_inference():
    runner = KerasRunner(model_name)


if __name__ == '__main__':
    pass

