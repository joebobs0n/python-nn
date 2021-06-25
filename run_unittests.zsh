#!/usr/bin/zsh

if [[ ! -f "$( realpath . )/run_unittests.zsh" ]]; then
    echo "\033[91m-F-\033[0m Change to \033[1mproject root\033[0m directory and try again..."
    exit 1
fi

tests=(
    neuron
)

foreach test in $tests
    fname=$( basename $test .py )
    $( which python3 ) "src/_ut_${fname}.py" || exit 1
    echo
end
