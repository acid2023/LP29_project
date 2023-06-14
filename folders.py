import os

logs_folder = '/logs'
models_folder = '/models'


def folder_check(folder: str) -> str:
    current_dir = os.getcwd()
    relative_path = f'{current_dir}{folder}'
    if not os.path.exists(relative_path):
        os.makedirs(relative_path)
    return relative_path + '/'


if __name__ == "__main__":
    pass
