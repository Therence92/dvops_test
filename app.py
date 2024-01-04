from flask import Flask

app = Flask(__name__)

@app.route('/')
def Hello_world() :
    
    return  ('Hello, My Name is Thierry')
      #print ('Welcome in my word')
  
if __name__=='__main__':
    app.run(host='0.0.0.0',port=80)