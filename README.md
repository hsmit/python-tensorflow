# python-tensorflow
onderdeel van knutseldag

## installatie voor tensorflow

Download python 3.6.2 64-bit van https://www.python.org/downloads/release/python-362/
Doe dit met PATH variabele en PIP!

Check python version op command-line: `python --version`.

Check ook pip version op command-line: `pip3 --version`.

Installeer tensorflow met pip: `pip3 install --upgrade tensorflow`.

Test installatie

    import tensorflow as tf 
    
    hello = tf.constant('Hello, TensorFlow!') 
    sess = tf.Session() 
    print(sess.run(hello)) 