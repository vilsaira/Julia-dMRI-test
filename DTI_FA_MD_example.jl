using NIfTI
using DelimitedFiles
using LinearAlgebra
using Plots

# Change to whatever dMRI data available
ni = niread("/mnt/g/dMRI_datasets/HCP/103818_dMRI_only/103818/T1w/Diffusion/slice.nii.gz");
bval = readdlm("/mnt/g/dMRI_datasets/HCP/103818_dMRI_only/103818/T1w/Diffusion/bvals");
bvec = readdlm("/mnt/g/dMRI_datasets/HCP/103818_dMRI_only/103818/T1w/Diffusion/bvecs");

function GetDesignMatrix_DT(bval, bvec)
    ngrad = length(bval)
    X = ones(ngrad, 7);
    for i in 1:ngrad
        b = bval[i]
        g = bvec[:,i]
        X[i,2] = -b*g[1]^2; # xx
        X[i,3] = -b*2*g[1]*g[2]; # xy
        X[i,4] = -b*2*g[1]*g[3]; # xz
        X[i,5] = -b*g[2]^2; # yy
        X[i,6] = -b*2*g[2]*g[3]; # yz
        X[i,7] = -b*g[3]^2; # zz
    end;
    return X;
end;

function GetEigsFromBetas(beta)
    dims = size(beta);
    nvox = dims[2];
    dt = zeros(3,3);
    evals = zeros(nvox,3);
    evecs = zeros(nvox,3,3);
    for i in 1:nvox
        if any(isnan.(beta[:,i]))
            continue
        end;
        dt[1,1] = beta[2,i]; # xx
        dt[1,2] = beta[3,i]; # xy
        dt[1,3] = beta[4,i]; # xz
        dt[2,1] = beta[3,i]; # xy
        dt[2,2] = beta[5,i]; # yy
        dt[2,3] = beta[6,i]; # yz
        dt[3,1] = beta[4,i]; # xz
        dt[3,2] = beta[6,i]; # yz
        dt[3,3] = beta[7,i]; # zz
        evals[i,:] = eigvals(dt);
        evecs[i,:,:] = eigvecs(dt);
    end;
    return evals, evecs;
end;

X = GetDesignMatrix_DT(bval, bvec);
# Log-linearize dwis
dims = size(ni);
nvox = dims[1]*dims[2]*dims[3];
dwi=reshape(ni, (nvox,dims[4]));
logdwi = log.(dwi);
# Calculate tensor elements using OLLS estimator
beta = X\logdwi';
# Calculate eigenvalues
evals, evecs = GetEigsFromBetas(beta);

# Calculate MD, FA from eigenvalues
MD = sum(evals, dims=2)/3;
l1=evals[:,1]
l2=evals[:,2]
l3=evals[:,3]
FA = sqrt.( (l1-l2).^2 + (l1-l3).^2 + (l2-l3).^2 ) ./ sqrt.(l1.^2+l2.^2+l3.^2) ./ sqrt(2);

# Reshape back into 3D volumes
FA = reshape(FA, dims[1:3]);
MD = reshape(MD, dims[1:3]);

# Save nifti images, note: header is incorrect so the result won't fit on DWI
# volume. Needs to be solved, how to copy header information correctly.
niwrite("MD.nii.gz", NIVolume(MD))
niwrite("FA.nii.gz", NIVolume(FA))