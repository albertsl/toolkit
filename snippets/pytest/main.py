def suma(x, y):
    return x+y

def filewrite(file, text):
    with open(file, "w") as f:
        f.write(text)