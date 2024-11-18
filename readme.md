# Yifan Yang's Personal Website

## References

Theme and customization based on https://github.com/pages-themes/minimal

Dark mode support based on https://github.com/godalming123/minimal and https://github.com/pages-themes/minimal/pull/121/files

Icons in author info section from https://fontawesome.com/v4.7/examples/

Repo organized as https://github.com/academicpages/academicpages.github.io

Latex support based on https://github.com/ryanmcdermott/jekyll-latex

## Repo Structure

`_layouts` describes the (customized) layout of the HTML page  
`index.md` contains the main text  
`_config.yml` contains configuration parameters for the site  
`_sass` contains styling settings
- for font awesome, you need to change the `fa-font-path` in `_variables.scss` to the path where you store the icons
- other files are for custom color scheme (e.g. dark mode) and element size in the page

`assets/fonts` contains all icons from font awesome  
`assets/css` imports the font awesome style and color scheme to the site, then you can use the icon and color scheme (examples in `_layouts`)