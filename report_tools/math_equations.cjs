const {mathjax}=require('./math/node_modules/mathjax-full/js/mathjax.js');
const {TeX}=require('./math/node_modules/mathjax-full/js/input/tex.js');
const {SVG}=require('./math/node_modules/mathjax-full/js/output/svg.js');
const {liteAdaptor}=require('./math/node_modules/mathjax-full/js/adaptors/liteAdaptor.js');
const {RegisterHTMLHandler}=require('./math/node_modules/mathjax-full/js/handlers/html.js');
const {AllPackages}=require('./math/node_modules/mathjax-full/js/input/tex/AllPackages.js');
const sharp=require('sharp');
const adaptor=liteAdaptor();RegisterHTMLHandler(adaptor);
const doc=mathjax.document('',{InputJax:new TeX({packages:AllPackages}),OutputJax:new SVG({fontCache:'local'})});
const equations={
fer_loss:String.raw`L=\lambda L_{\mathrm{focal}}(z,y_a)+(1-\lambda)L_{\mathrm{focal}}(z,y_b)+0.01L_{\mathrm{center}}`,
box_center:String.raw`x=\frac{x_{\mathrm{left}}+x_{\mathrm{right}}}{2},\qquad y=\frac{y_{\mathrm{top}}+y_{\mathrm{bottom}}}{2}`,
rmse:String.raw`\mathrm{RMSE}=\sqrt{\frac{1}{2N}\sum_{i=1}^{N}\left[(\hat x_i-x_i)^2+(\hat y_i-y_i)^2\right]}`,
vessel_loss:String.raw`L=0.4L_{\mathrm{BCE}}+0.5L_{\mathrm{Dice}}+0.1rL_{\mathrm{clDice}}+L_{\mathrm{aux}}`,
dice:String.raw`\mathrm{Dice}=\frac{2|P\cap G|}{|P|+|G|}`
};
(async()=>{for(const [name,tex]of Object.entries(equations)){
 let svg=adaptor.outerHTML(doc.convert(tex,{display:true}));svg=svg.slice(svg.indexOf('<svg'),svg.lastIndexOf('</svg>')+6);
 svg=svg.replace(/(width|height)="([0-9.]+)ex"/g,(_,dim,n)=>`${dim}="${Math.ceil(Number(n)*22)}px"`).replace(/currentColor/g,'#000000');
 await sharp(Buffer.from(svg),{density:180}).png().toFile(`work/figures/${name}.png`);
 console.log(name);
}})();
