# Bootstrap basic structure
Bootstrap divides the space in 12 blocks of the same size

Botstrap works with different screen sizes:

| Breakpoint Class | Infix | Dimensions |
|------------------|-------|------------|
| X-Small          | *None* | <576px     |
| Small            | `sm`   | ≥576px     |
| Medium           | `md`   | ≥768px     |
| Large            | `lg`   | ≥992px     |
| Extra large      | `xl`   | ≥1200px    |
| Extra extra large| `xxl`  | ≥1400px    |

Screen sizes have priority from smaller to larger. If we only specify values for `sm` and `xl`, on a `md` size screen, it will use the same value as in `sm` screen


Generally, we will create a main element called `container`:

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

# Margin and Padding
Margin and padding can be applied using  `{property}{sides}-{size}` for `xs` and `{property}{sides}-{breakpoint}-{size}` for `sm`, `md`, `lg`, `xl`, and `xxl`.

Where property is one of:

    m - for classes that set margin
    p - for classes that set padding

Where sides is one of:

    t - for classes that set margin-top or padding-top
    b - for classes that set margin-bottom or padding-bottom
    s - (start) for classes that set margin-left or padding-left in LTR, margin-right or padding-right in RTL
    e - (end) for classes that set margin-right or padding-right in LTR, margin-left or padding-left in RTL
    x - for classes that set both *-left and *-right
    y - for classes that set both *-top and *-bottom
    blank - for classes that set a margin or padding on all 4 sides of the element

Where size is one of:

    0 - for classes that eliminate the margin or padding by setting it to 0
    1 - (by default) for classes that set the margin or padding to $spacer * .25
    2 - (by default) for classes that set the margin or padding to $spacer * .5
    3 - (by default) for classes that set the margin or padding to $spacer
    4 - (by default) for classes that set the margin or padding to $spacer * 1.5
    5 - (by default) for classes that set the margin or padding to $spacer * 3
    auto - for classes that set the margin to auto


# Cheatsheet
A cheatsheet with examples of what can be done with Bootstrap can be found [here](https://getbootstrap.com/docs/5.0/examples/cheatsheet/)