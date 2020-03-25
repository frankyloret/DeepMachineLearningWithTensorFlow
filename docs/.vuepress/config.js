module.exports = {
  title: 'DeepMachineLearningWithTensorFlow',
  description: 'Deep Machine Learning with TensorFlow',
  themeConfig: {
    nav: [
      { text: 'Home', link: '/' },
      { text: 'VIVES', link: 'https://www.vives.be' },
      { text: 'License', link: '/LICENSE.md' },
    ],
    sidebar: [
      ['/', 'Home'],
      ['/first-look-at-tensorflow/', 'First Look at TensorFlow'],
      ['/tensorflow-on-rpi4/', 'Tensorflow_on_RPi4']
    ],
    repo: 'https://github.com/frankyloret/DeepMachineLearningWithTensorFlow',
    docsDir: 'docs',
    docsBranch: 'master'
  },
  markdown: {
    lineNumbers: true,
  },
  serviceWorker: true,
  plugins: [
    ['vuepress-plugin-zooming', {
      // selector for images that you want to be zoomable
      // default: '.content img'
      selector: 'img',

      // make images zoomable with delay after entering a page
      // default: 500
      // delay: 1000,

      // options of zooming
      // default: {}
      options: {
        bgColor: 'black',
        zIndex: 10000,
      },
    }],
  ],
}
