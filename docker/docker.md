Tutorial for Docker: 

https://docker-curriculum.com

Run a container with:

`docker run xxxxx`
 
Once containers are closed, run this command to clean them:

`docker container prune`

Images are the bluprint of the application
Containers are instances of the images. They run the application. They are created using `docker run xxxx`

When using `docker run xxxx`, we can add `--rm` and `-it` flags. 

`--rm` automatically removes the container once exited. 

`-it` creates an interactive terminal.

Example:

`docker run --rm -it xxxx`

Other important parameters are:

`-d` runs the container detached from our terminal. We can close the window and the container will keep running

`-P` will publish all ports. We can see them

`--name yyyy` Gives a name to the container

`docker port yyyy`  Shows the available ports
 
We can specify a custom port to which the client will forward connections to the container with:

`docker run -p 3500:80 xxxx `

To stop a detached container, run `docker stop yyyy`.  `yyyy` can be the container name or the container ID

A DOCKERFILE is a text file that contains a list of commands that Docker calls while creating an image.

There is an example DOCKERFILE in this folder. It corresponds to a webapp written in Python with Flask.

We have to build our image from the DOCKERFILE with this command:

`docker build .`

where . is the folder. We can also give it a name:

`docker build -t name .`

with `docker images` we can see if the image has been created

we can run a container with:

`docker run -p 8888:5000 name`

We can now access the website from `localhost:8888`