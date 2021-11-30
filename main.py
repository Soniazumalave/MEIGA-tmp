#!/usr/bin/env python
#coding: utf-8

""" 
MEIGA-SR:
Program to detect retrotransposon integrations from pair end sequencing data
"""

if __name__ == '__main__':
	
	VERSION = '1.1.0'
	
	####################
	## Import modules ##
	####################
 
	## 1. Check program dependencies are satisfied 
	################################################
	from GAPI import check_dependencies as cd
	from GAPI import log

	missingDependencies = cd.missing_python_dependencies() or cd.missing_program_dependencies()
	if missingDependencies: exit(1)

	# External
	import argparse
	import sys
	import os
	import multiprocessing as mp
	import configparser
	import time
	from datetime import datetime
	import logging

	# Internal
	from GAPI import bamtools
	from modules import caller

	######################
	## Get user's input ##
	######################
 
	## 0. Set timer
	##################
	start = time.time()
  

	## 1. Define parser
	######################
	parser = argparse.ArgumentParser()

	### Define subcommands 
	subparsers = parser.add_subparsers(title='subcommands')
	call = subparsers.add_parser('call', help='Call mobile element insertions from short-read data', description='Call mobile element insertions from short-read data. Two running modes: 1) SINGLE: individual sample; 2) PAIRED: tumour and matched normal sample')
	call.set_defaults(call_tds=False)
	call_tds = subparsers.add_parser('call-tds', help='Call mobile element transductions from short-read data', description='Call transductions for a set of target loci from target or whole genome sequencing data. Two running modes: 1) SINGLE: individual sample; 2) PAIRED: tumour and matched normal sample')
	call_tds.set_defaults(call_tds=True)

	### "call" mode: 
	# A. Mandatory arguments
	call.add_argument('config', help='Configuration file')
	call.add_argument('bam', help='Input bam file. Will correspond to the tumour sample in the PAIRED mode')
	# B. Optional arguments
	call.add_argument('--normalBam', default=None, dest='normalBam', help='Matched normal bam file. If provided MEIGA will run in PAIRED mode')
	call.add_argument('-o', '--outDir', default=os.getcwd(), dest='outDir', help='Output directory. Default: current working directory')
	call.add_argument('-p', '--processes', default=1, dest='processes', type=int, help='Number of processes. Default: 1')
	call.add_argument('-d', '--debug', action='store_true', dest='debug', help='Debug mode')
	call.add_argument('--predict', action='store_true', dest='predict', help='Apply ML classifier to output')

	### "call-tds" mode: 
	# A. Mandatory arguments
	call_tds.add_argument('config', help='Configuration file')
	call_tds.add_argument('bam', help='Input bam file. Will correspond to the tumour sample in the PAIRED mode')
	# B. Optional arguments
	call_tds.add_argument('--normalBam', default=None, dest='normalBam', help='Matched normal bam file. If provided MEIGA will run in PAIRED mode')
	call_tds.add_argument('-o', '--outDir', default=os.getcwd(), dest='outDir', help='Output directory. Default: current working directory')
	call_tds.add_argument('-p', '--processes', default=1, dest='processes', type=int, help='Number of processes. Default: 1')
	call_tds.add_argument('-d', '--debug', action='store_true', dest='debug', help='Debug mode')

	### Set method features
	scriptName = os.path.basename(sys.argv[0])
	scriptName = os.path.splitext(scriptName)[0]

	mp.set_start_method('spawn')


	## 2. Parse user's input
	##########################
	args = parser.parse_args()
	configFile = args.config
	bam = args.bam
	normalBam = args.normalBam
	processes = args.processes
	
	
	## 2.1. Set output dir
	##########################
	outDir = args.outDir

	# if debug, use outDir/timeStamp as the output directory
	if args.debug:
		timeStamp = str(datetime.now().strftime("%Y%m%d%H%M%S"))
		outDir = args.outDir + '/' + timeStamp

	# create output dir
	os.makedirs(outDir, exist_ok=True)

  
	## 2.2. Determine running mode
	##################################
	mode = 'PAIRED' if normalBam else 'SINGLE'
	
  
	## 2.3. Create configuration dictionary
	###########################################
	confDict = {}
	meigaConfig = configparser.ConfigParser(inline_comment_prefixes = ('#'))
	meigaConfig.read(configFile)
	config = meigaConfig['MEIGA-SR']

	### General
	reference = config.get('reference')
	refDir = config.get('refDir')
	confDict['source'] = 'MEIGA-SR-' + VERSION
	confDict['species'] = config.get('species')
	confDict['build'] = config.get('build')
	confDict['annovarDir'] = config.get('annovarDir')
	confDict['germlineMEI'] = None if config.get('germlineMEI') == 'none' else config.get('germlineMEI')
	confDict['processes'] = processes
	confDict['debug'] = args.debug
	confDict['predict'] = args.predict

	### BAM processing
	confDict['targetBins'] = None if config.get('targetBins') == 'none' else config.get('targetBins')
	confDict['binSize'] = config.getint('binSize')
	confDict['filterDup'] = config.getboolean('noDuplicates')
	confDict['readFilters'] = [filt.strip() for filt in config.get('readFilters').split(',')]
	confDict['minMAPQ'] = config.getint('minMAPQ')
	confDict['minCLIPPINGlen'] = config.getint('minCLIPPINGlen')
	confDict['targetEvents'] = ['DISCORDANT', 'CLIPPING']

	### Target refs
	refs = config.get('refs')
	if refs == 'ALL': refs = bamtools.get_refs(bam) # If "ALL" specified, get all refs in bam file
	targetRefs = [ref.strip() for ref in refs.split(',')]
	confDict['targetRefs'] = targetRefs

	### Clustering
	confDict['minClusterSize'] = config.getint('minClusterSize')
	confDict['maxClusterSize'] = config.getint('maxClusterSize')
	confDict['maxBkpDist'] = config.getint('BKPdist')
	confDict['minPercRcplOverlap'] = config.getint('minPercOverlap')
	confDict['equalOrientBuffer'] = config.getint('equalOrientBuffer')
	confDict['oppositeOrientBuffer'] = config.getint('oppositeOrientBuffer')

	### Filtering thresholds
	confDict['minReads'] = config.getint('minReads')
	confDict['minNormalReads'] = config.getint('minNormalReads')
	confDict['minNbDISCORDANT'] = config.getint('minClusterSize')
	confDict['minNbCLIPPING'] = config.getint('minClusterSize')
	confDict['minReadsRegionMQ'] = config.getfloat('minReadsRegionMQ')
	confDict['maxRegionlowMQ'] = config.getfloat('maxRegionlowMQ')
	confDict['maxRegionSMS'] = config.getfloat('maxRegionSMS')

	### Transduction search
	confDict['retroTestWGS'] = config.getboolean('wgsData')
	confDict['blatClip'] = config.getboolean('blatClip')
	confDict['tdEnds'] = [tdEnd.strip() for tdEnd in config.get('transductionEnds').split(',')]
	confDict['srcBed'] = None if config.get('sourceBed') == 'none' else config.get('sourceBed')
	confDict['srcFamilies'] = [family.strip() for family in config.get('srcFamilies').split(',')]

	# In debug mode the output is the specified dir + data_time
	if args.debug:
		outDir = args.outDir + "/"+str( datetime.now().strftime("%Y%m%d%H%M%S") ) 
	else:
		outDir = args.outDir

	# create  output and log dirs
	logDir = outDir + '/logs'
	confDict['outDir'] = outDir
	confDict['logDir'] = logDir

	os.makedirs(outDir, exist_ok=True)
	os.makedirs(logDir, exist_ok=True)


	## 2.5 Initialize log system
	##############################
	logName = 'main'
	logFile = logDir + '/main.log'
	logFormat = '%(asctime)s - %(name)s - %(levelname)s - %(message)s' 
	logger = log.setup_logger(logName, logFile, logFormat, level=logging.DEBUG, consoleLevel=logging.WARNING)
        

	## 3. Display configuration to standard output
	################################################
	logger.info('***** ' + scriptName + ' ' + VERSION + ' configuration *****')
	logger.info('*** Arguments ***')
	subcommand = 'call-tds' if args.call_tds else 'call'
	logger.info('subcommand: ' + subcommand)
	logger.info('config file: ' + configFile)
	logger.info('bam: '+  bam)
	logger.info('normalBam: ' + str(normalBam))
	logger.info('outDir: ' + outDir)
	logger.info('processes: ' + str(processes) + '\n\n')
	logger.info('*** ConfigDict ***')
	for key, value in confDict.items():
		logger.info(key + ' => ' + str(value))
	logger.info('\n\n')

  
	## 4. Check all required DB are located in the dirs provided
	##############################################################
	missingDB = cd.missing_db(refDir, confDict['annovarDir'])
	if missingDB: exit(1)

	
	########################
	## Execute MEI caller ##
	########################
	
	# If 'call-tds' running mode selected
	if args.call_tds:
		# execute transductions caller
		meiCaller = caller.transduction_caller(mode, bam, normalBam, reference, refDir, confDict)

	else:
		# execute universal MEI caller
		meiCaller = caller.MEI_caller(mode, bam, normalBam, reference, refDir, confDict)

	meiCaller.call()


	############
	##  Exit  ##
	############
	
	timeCounter = round((time.time()-start)/60, 4)
	logger.info('***** Finished! in ' + str(timeCounter) + 'minutes *****\n')


