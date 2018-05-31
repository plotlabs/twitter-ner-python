##Organization Name Recogntion from Tweets

This repository contains python code which detects an Organization's name from a list of tweets extracted using the [Algorithmia API](https://github.com/algorithmiaio/algorithmia-python) .

The main executable file is ner_main.py. A python dictionary is input to the function pull_tweets(), with the dictionary keys set as:

```javascript
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

