##Organization Name Recogntion from Tweets

This repository contains python code which detects an Organization's name from a list of tweets extracted using the [Algorithmia API](https://github.com/algorithmiaio/algorithmia-python) .

The main executable file is ner_main.py. A python dictionary is input to the function pull_tweets(), with the dictionary keys set as:

```python
{
	"query": Search Keyword/Organization Name 
	"numTweets": Number of Tweets
	"auth": Algorithmia API authorization Token
	"app_key": Twitter Application Key 
	"app_secret": Twitter Application Secret Token
	"oauth_token": Twitter Application Authorization Token
	"oauth_token_secret": Twitter Application Authorization Secret Token
}
```

The python file can be executed from the terminal via 

```shell
python ner_main.py
```

### Test Case

for input query:
```python
input = {
        "query": "Google",
        "numTweets": "700",
        "auth": {
            "app_key": 'your_consumer_key',
            "app_secret": 'your_consumer_secret_key',
            "oauth_token": 'your_access_token',
            "oauth_token_secret": 'your_access_token_secret'
        }
    }


```

The following output is obtained:

```python

[{

      'ORGANIZATION':Counter(      {  
         'Nations':6,
         'First':6,
         'Services':2,
         'TV':2,
         'Oils':1,
         'Soup':1,
         'Zenit':1,
         'Northwest':1,
         'Philip':1,
      }      )
   }
]

```
