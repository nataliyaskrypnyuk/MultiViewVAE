# MultiViewVAE
Exploration of a latent space of a Variational Autoencoder trained on multiple views of the same 3D object - a mesh of the scanned jaw

Needs a folder with dataset cases having the following structure: MainFolder->CaseId->DataFiles->{set of pictures}

Set of pictures should look like this example: <img width="1688" height="859" alt="image" src="https://github.com/user-attachments/assets/6a98f008-31ad-4dd6-ab94-7733c965b9aa" />

It can also include only pictures for the upper jaw or only pictures for the lower jaw, i.e. include 3 pictures per case instead of 6 as shown above

To train a single-view VAE, run: python train_singleview_vae.py

To train a multi-view VAE, run: python train_multiviw_vae.py

To view the experiments' loss, run "mlflow ui" and open "http://127.0.0.1:5000/"

ToDo: add environment, add different schedules for beta

