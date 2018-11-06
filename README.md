# Detección de DNI

Este programa se ha creado utilizando las librerías OpenCV. Emplea la cámara para detectar el DNI y de reconocer la cara.

1. Antes de que el programa pueda detectar el DNI necesita unas imágenes de referencia para poder reconocerlo en escena.

2. Durante el funcionamiento del programa se buscará en escena si hay un DNI utilizando el algoritmo SURF, encontrando semejanzas entre las imágenes de referencia y los elementos de la escena. Funciona incluso si el DNI está boca abajo:

![detectagithub](https://user-images.githubusercontent.com/44776831/48088698-f8654b00-e202-11e8-8c17-64fd4022c51e.png)

3. Además, el programa puede detectar y extraer del DNI la cara, además de la firma y el MRZ de la parte trasera:

![facegithub](https://user-images.githubusercontent.com/44776831/48088837-4ed28980-e203-11e8-937d-3668e9bb1b8c.png)
