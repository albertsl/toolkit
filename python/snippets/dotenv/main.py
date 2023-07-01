from dotenv import load_dotenv, dotenv_values
from os import getenv

load_dotenv() # If no path is specified, the default is .env
print(getenv("API_KEY"))

env_vars = dotenv_values()
print(env_vars["API_KEY"])