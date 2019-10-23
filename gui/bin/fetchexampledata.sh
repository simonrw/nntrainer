#!/bin/sh

set -e

main() {
    mkdir -p exampledata/{train,val}/{cat,dog}

    # Cats
    i=0
    for image in \
        "https://en.wikipedia.org/wiki/Cat_intelligence#/media/File:An_up-close_picture_of_a_curious_male_domestic_shorthair_tabby_cat.jpg" \
        "https://images.immediate.co.uk/production/volatile/sites/4/2018/08/iStock_000044061370_Medium-fa5f8aa.jpg?quality=90&crop=5px,17px,929px,400px&resize=940,404" \
        ; do
        for subdir in exampledata/train/cat exampledata/val/cat; do
            outpath="${subdir}/${i}.jpg"
            test -e $outpath || curl -Lo $outpath $image
        done
        i=$((i+1))
    done


    # Dogs
    i=0
    for image in \
        "https://www.petmd.com/sites/default/files/senior-golden-retriever-with-ball-picture-id488657289.jpg" \
        "https://s3.amazonaws.com/cdn-origin-etr.akc.org/wp-content/uploads/2017/11/12234558/Chinook-On-White-03.jpg" \
        ; do
        for subdir in exampledata/train/dog exampledata/val/dog; do
            outpath="${subdir}/${i}.jpg"
            test -e $outpath || curl -Lo $outpath $image
        done
        i=$((i+1))
    done

}

main
