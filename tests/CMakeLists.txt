# @file CMakeLists.txt
# Tests of the SCALAR tagger, one ctest test per behavior. The golden files in
# data/ hold the answers of the Python SCALAR, written by the Python scripts of this folder (PORT.md).

set(SCALAR_GOLDEN "${CMAKE_CURRENT_SOURCE_DIR}/data")

# The Ronin splitter and its Snowball stemmer
add_executable(scalarSplitterTests splitterTests.cpp escapedLines.hpp)
target_link_libraries(scalarSplitterTests PRIVATE scalar)
target_compile_options(scalarSplitterTests PRIVATE ${SCALAR_WARNINGS})
foreach(test IN ITEMS
        stemmerMatchesNltkOnGoldenWords
        splitMatchesPythonOnGoldenIdentifiers
        splitGivesTheGoldenTokensOnEightThreadsAtOnce
        longestTermWithNumbersWinsWhereTermsCompete
        termPatternsKeepTheirRegexMeaning
        identifiersWithoutTokensGiveNoTokens
        unsupportedTermSyntaxIsRejected
        loadingAFolderWithoutTablesThrows
        platformPowMatchesPythonMathPowForEveryRescaledCount)
    add_test(NAME scalar.${test}
             COMMAND scalarSplitterTests ${test} "${SCALAR_RONIN_PATH}" "${SCALAR_GOLDEN}/splitter")
endforeach()

# The model input: feature tokens, position tokens, and WordPiece
add_executable(scalarEncoderTests encoderTests.cpp caseFormat.cpp caseFormat.hpp)
target_link_libraries(scalarEncoderTests PRIVATE scalar)
target_compile_options(scalarEncoderTests PRIVATE ${SCALAR_WARNINGS})
foreach(test IN ITEMS
        tokenizerLoadsTheVocabularyOfTheModelFolder
        tokenizerRejectsSettingsThatThePortDoesNotFollow
        featureListsNormalizeAsInPython
        modelWithoutSavedFeaturesUsesTheLegacyFeatures
        sequenceMatcherRatioEqualsDifflib
        pythonSumEqualsTheBuiltInSum
        stressCasesEncodeAsInPython
        corpusRowsEncodeAsInPythonInThreeFeatureSets
        taggerRowsEncodeAsInPythonWithTheirOwnTypeLanguageAndSystem
        wordsWithoutASubtokenThrowIdentifierTooLongError
        bytesAboveAsciiAreRejected)
    add_test(NAME scalar.${test}
             COMMAND scalarEncoderTests "${SCALAR_GOLDEN}/encoder" "${SCALAR_DATA_PATH}" ${test})
endforeach()

# DistilBERT, the CRF, the safetensors reader, and the reader of built-in files
add_executable(scalarModelTests modelTests.cpp goldenData.cpp goldenData.hpp)
target_link_libraries(scalarModelTests PRIVATE scalar)
target_compile_options(scalarModelTests PRIVATE ${SCALAR_WARNINGS})
foreach(test IN ITEMS
        dataFilesViewBuiltInFilesInPlace
        dataFilesReadFilesOnDiskWhole
        safetensorsReadsValidTensors
        safetensorsReadsBuiltInTensors
        safetensorsRejectsMalformedFiles
        crfMatchesTorchcrfOnSyntheticEmissions
        crfDecodesPythonEmissionsToPythonLabels
        kernelsMatchTorchBitForBit
        emulatedFmaEqualsStdFmaOnHardCases
        linearLayerFollowsTheChainOrderOnEveryShape
        modelRejectsInvalidInputs
        modelReturnsNoLabelsWithoutWords
        modelGivesPythonLabelsOnGoldenRows
        modelEmissionsStayCloseToPython
        decodeBatchEqualsDecodeBitForBit)
    add_test(NAME scalar.${test}
             COMMAND scalarModelTests ${test} "${SCALAR_DATA_PATH}" "${SCALAR_GOLDEN}/model")
endforeach()

# The plain kernels of SCALAR_PORTABLE_KERNELS and a Debug build run the model tests for minutes
set_tests_properties(scalar.modelGivesPythonLabelsOnGoldenRows scalar.decodeBatchEqualsDecodeBitForBit
                     scalar.modelEmissionsStayCloseToPython PROPERTIES TIMEOUT 3600)

# The whole tagger
add_executable(scalarTaggerTests scalarTaggerTests.cpp)
target_link_libraries(scalarTaggerTests PRIVATE scalar)
target_include_directories(scalarTaggerTests PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_compile_definitions(scalarTaggerTests PRIVATE "SCALAR_TEST_WORDS=\"${SCALAR_GOLDEN}/tagger/custom_words.txt\"" ${SCALAR_FOLDER_DEFINITIONS})
target_compile_options(scalarTaggerTests PRIVATE ${SCALAR_WARNINGS})
foreach(test IN ITEMS
        taggerGivesThePythonAnswersOnGoldenNames
        tagAllGivesTheAnswersOfTagOnEveryThreadCount
        dictionaryFlagFollowsTheWordListsOfTheServer
        namesWithoutWordsAreTaggedWithNoWords
        namesTooLongForTheModelGetItsError
        longRunsOfOneCaseSplitOnASmallStack
        wordsToJsonGivesTheBodyOfTheServer
        builtInFilesNameOnlyFilesInTheBinary)
    add_test(NAME scalar.${test}
             COMMAND scalarTaggerTests ${test} "${SCALAR_GOLDEN}/tagger")
    set_tests_properties(scalar.${test} PROPERTIES TIMEOUT 3600)
endforeach()
