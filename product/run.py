#!/usr/bin/env python
#from flaskexample2 import app
#from mvp import app
from mbp import app
#app.run(debug = True) # for running locally
app.run(host='0.0.0.0', debug=True) # for the website
