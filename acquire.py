"""
A module for obtaining repo readme and language data from the github API.
Before using this module, read through it, and follow the instructions marked
TODO.
After doing so, run it like this:
    python acquire.py
To create the `data.json` file that contains the data.
"""

import pandas as pd
import os
import time
import csv
import json
from typing import Dict, List, Optional, Union, cast
import requests

from bs4 import BeautifulSoup

from env import github_token, github_username

# Check if repo csv exists
#if not os.path.isfile("repo.csv"):
    
    
lang_list = ['JavaScript', 'HTML', 'Java', 'Python', 'CSS', 'C#', 'PHP', 'TypeScript', 'R']
#     page_num = [1-23]
repos = []

#     for page in page_num:
for i in range(1,41):

#             url = 'https://github.com/search?l={lang}&p={i}&q=mental+health&type=repositories'
    url = f"https://github.com/search?p={i}&q=%23mental-health&type=Repositories"
    while True:
        response = requests.get(url)
        if response.ok:
            print(True)
            break
        else:
            print('sleeping')
            time.sleep(20)
            
    soup = BeautifulSoup(response.content, 'html.parser')

    for element in soup.find_all('a', class_='v-align-middle'):
        repos.append(element.text)

#         time.sleep(10)

with open('repo.csv', 'w') as createfile:
    wr = csv.writer(createfile, quoting=csv.QUOTE_ALL)
    wr.writerow(repos)
    results = []
with open('repo.csv', newline='') as inputfile:
    results = list(csv.reader(inputfile))

    REPOS = [item for sublist in results for item in sublist]
    headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}

if headers["Authorization"] == "token " or headers["User-Agent"] == "":
    raise Exception(
    "You need to follow the instructions marked TODO in this script before trying to use it"
)


def github_api_request(url: str) -> Union[List, Dict]:
    response = requests.get(url, headers=headers)
    response_data = response.json()
    if response.status_code != 200:
        raise Exception(
            f"Error response from github api! status code: {response.status_code}, "
            f"response: {json.dumps(response_data)}"
        )
    return response_data


def get_repo_language(repo: str) -> str:
    url = f"https://api.github.com/repos/{repo}"
    repo_info = github_api_request(url)
    if type(repo_info) is dict:
        repo_info = cast(Dict, repo_info)
        return repo_info.get("language", None)
    raise Exception(
        f"Expecting a dictionary response from {url}, instead got {json.dumps(repo_info)}"
    )
    
def get_repo_contents(repo: str) -> List[Dict[str, str]]:
    url = f"https://api.github.com/repos/{repo}/contents/"
    contents = github_api_request(url)
    if type(contents) is list:
        contents = cast(List, contents)
        return contents
    raise Exception(
        f"Expecting a list response from {url}, instead got {json.dumps(contents)}"
    )

def get_readme_download_url(files: List[Dict[str, str]]) -> str:
    """
    Takes in a response from the github api that lists the files in a repo and
    returns the url that can be used to download the repo's README file.
    """
    for file in files:
        if file["name"].lower().startswith("readme"):
            return file["download_url"]
    return ""

def process_repo(repo: str) -> Dict[str, str]:
    """
    Takes a repo name like "gocodeup/codeup-setup-script" and returns a
    dictionary with the language of the repo and the readme contents.
    """
    contents = get_repo_contents(repo)
    readme_download_url = get_readme_download_url(contents)
    if readme_download_url == "":
        readme_contents = None
    else:
        readme_contents = requests.get(readme_download_url).text
    return {
        "repo": repo,
        "language": get_repo_language(repo),
        "readme_contents": readme_contents,
    }

def scrape_github_data() -> List[Dict[str, str]]:
    """
    Loop through all of the repos and process them. Returns the processed data.
    """
    return [process_repo(repo) for repo in REPOS]


#if __name__ == "__main__":
data = scrape_github_data()
json.dump(data, open("data.json", "w"), indent=1)