import datetime

def file_str():
    """Generates a datetime string in a good format to save as a file

    Returns:
        str: String with the datetime in a good format to save as a file
    """
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if __name__ == '__main__':
    # Save a file using a good formatted datetime string
    with open(f"{dt}.txt", 'w') as save_file:
        # Data can be logged with the datetime moment it was captured
        save_file.write(file_str() + " " + str(5) + '\n')