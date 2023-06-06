Bootstrap divides the space in 12 blocks of the same size

Botstrap works with 4 different screen sizes:

- sm: small
- md: medium
- lg: large
- xl: extra large

Generally we will create a main element called `container`:

```
<div class="container">
</div>
```

Inside of the container we can have several `rows`:

```
<div class="container">
    <div class="row">
    </div>
    <div class="row">
    </div>
    <div class="row">
    </div>
</div>
```

We can divide each row into different columns

```
<div class="container">
    <div class="row">
        <div class="col-sm-12 col-md-4 col-lg-4 col-xl-4">
            <h1>Hello, world!</h1>
        </div>
        <div class="col-sm-12 col-md-4 col-lg-4 col-xl-4">
            <h1>Hello, world!</h1>
        </div>
        <div class="col-sm-12 col-md-4 col-lg-4 col-xl-4">
            <h1>Hello, world!</h1>
        </div>
    </div>
</div>
```