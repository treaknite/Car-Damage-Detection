const inpFile = document.getElementById('image');
const previewContainer = document.getElementById('imagePreview');
const previewImage = previewContainer.querySelector('.image-preview__image');
const previewDefaultText = previewContainer.querySelector('.image-preview__default-text');

//const { MongoClient } = require('mongodb');
//const uri = "mongodb+srv://datacrusade1999:btIOV51mRbm7ZDDa@cluster0.ccvdv.mongodb.net/myFirstDatabase?retryWrites=true&w=majority";
//const client = new MongoClient(uri, { useNewUrlParser: true, useUnifiedTopology: true });
//client.connect(err => {
//  const collection = client.db("test").collection("devices");
//  // perform actions on the collection object
//  client.close();
//});


// Image preview
inpFile.addEventListener('change', function() {
    const file = this.files[0];
    
    if (file) {
        const reader = new FileReader();
    
        previewDefaultText.style.display = 'none';
        previewImage.style.display = 'block';

        reader.addEventListener('load', function() {
            previewImage.setAttribute('src', this.result);
        });
        reader.readAsDataURL(file);
    } 

    else{
        previewDefaultText.style.display = null;
        previewImage.style.display = null;
        previewImage.setAttribute("src","");
    }
});

const about = document.getElementById('about-project');
about.addEventListener('click', function() {
    document.getElementById("playground").className = "nav-link";
    document.getElementById("about-project").className = "nav-link active";
    
});

const playground = document.getElementById('playground');
playground.addEventListener('click', function() {
    document.getElementById("about-project").className = "nav-link";
    document.getElementById("playground").className = "nav-link active";
});