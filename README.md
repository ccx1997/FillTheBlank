# FillTheBlank
To fill the blank in a given word or sentence.
The model is based on self attention (model parameter is given).
## An example
Command:
`python test.py --idc 0 --param param/attention2.pkl --word '神经?络'`

Output:
`('网', 0.403095006942749)  ('联', 0.15400202572345734)  ('交', 0.14229977130889893)`

The pretrained parameter file is at [Baidu Netdisk](https://pan.baidu.com/s/1IrPHcOmc30m9WuMRv2Veow). You can download it and put it in directory `param/`.
