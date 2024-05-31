import subprocess, sys

def run_make_run_in_directory(directory):
    task_name = ""
    if len(sys.argv) < 2:
        print("请指定测试项")
        return
    else:
        task_name = sys.argv[1]
    directory = task_name + "/" + directory
    cmd = f'cd {directory} && make run'
    # 使用subprocess.run执行命令，并捕获输出和错误
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # 输出结果
    if "bang" in directory:
        print("Bang Run")
    elif "sycl" in directory:
        print("SYCL Run")

    print(f"Directory: {directory}")
    #print("Output:")
    print(result.stdout)

directories = ['bangc', 'sycl']

for directory in directories:
    run_make_run_in_directory(directory)
