# micrography-imgpro
Image processing of micrography pictures.

# app.py / the easier one / for not informatic people
deploy : {todo there should be a link here}

or run:
```
streamlit run app.py
```

# Controller.py / cli interface

this is meant to be called from console, you can do:

``` 
python .\controller.py
```
the flags available are:
-f   to run getMeFibers
-fl  to run getMeFlashes
-p   to run getMetPores

example:

``` 
python .\controller.py -f -p
```
this will run both fibers and pores

running without flags will do getMeResults and output the colored images to a folder


# Dependencies

``` 
pip install streamlit opencv-python numpy scikit-image
``` 