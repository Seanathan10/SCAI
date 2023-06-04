from pytube import YouTube
from pytube import Playlist
from pytube.cli import on_progress
from pytube import exceptions
import threading
import multiprocessing
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from googleapiclient.discovery import build
import os
import csv
import array
import ffmpeg
from io import StringIO
import asyncio
from youtubesearchpython.__future__ import VideosSearch
import re

import pandas as pd

from audRead import audioMod

class Track():
	#                  genre,artist_name,track_name,track_id,popularity,acousticness,danceability,duration_ms,energy,instrumentalness,key,liveness,loudness,mode,speechiness,tempo,time_signature,valence
	def __init__( self, genre, artist_name, track_name, track_id, popularity, acousticness, danceability, duration_ms, energy, instrumentalness, key, liveness, loudness, mode, speechiness, tempo, time_signature, valence ):
		self.genre = genre
		self.artist_name = artist_name
		self.track_name = track_name
		self.track_id = track_id
		self.popularity = popularity
		self.acousticness = acousticness
		self.danceability = danceability
		self.duration_ms = duration_ms
		self.energy = energy
		self.instrumentalness = instrumentalness
		self.key = key
		self.liveness = liveness
		self.loudness = loudness
		self.mode = mode
		self.speechiness = speechiness
		self.tempo = tempo
		self.time_signature = time_signature
		self.valence = valence

	def name( self ):
		return self.track_name

	def __str__( self ):
		return f"{ self.track_name } by { self.artist_name }"
		# return f"{ self.track_name }"


async def HandleMP3( Link, SongName ):
	try:
		dl_file = YouTube( Link, on_progress_callback=on_progress )

		audio = dl_file.streams.get_audio_only()
		# save_dir = 'mp3s/'
		# save_dir = '.'

		
		print( f"({ audio.bitrate / 1000 } kbps) \"{ dl_file.title }\" downloading..." )

		outfile = audio.download()

		file_base, file_ext = os.path.splitext( outfile )
		# temp_string = f"{SongName}"
		cwd = os.getcwd()
		file_base = r"{}".format( SongName )
		# file_base.replace( "\\n", "" )
		file_base = file_base.strip('\n')
		file_base = file_base.strip('\t')
		final_file = cwd + "\\" + file_base + '.mp3'
		os.rename( outfile, final_file )
		
		# print( f"{ dl_file.title } downloaded" )
	except exceptions.AgeRestrictedError:
		print( f"{dl_file.title} is age restricted - SKIPPING" )
	except OSError:
		print( f"Error writing this song: {dl_file.title}"  )
	except exceptions:
		print( f"Some other YouTube error with song: {dl_file.title}" )



async def HandleYTMP3( Link ):
	try:
		dl_file = YouTube( Link, on_progress_callback=on_progress )

		audio = dl_file.streams.get_audio_only()
		# save_dir = 'mp3s/'
		# save_dir = '.'

		
		print( f"({ audio.bitrate / 1000 } kbps) \"{ dl_file.title }\" downloading..." )

		outfile = audio.download()

		file_base, file_ext = os.path.splitext( outfile )
		final_file = file_base + '.mp3'
		os.rename( outfile, final_file )
		
		# print( f"{ dl_file.title } downloaded" )
	except exceptions.AgeRestrictedError:
		print( f"{dl_file.title} is age restricted - SKIPPING" )
	except OSError:
		print( f"Error writing this song: {dl_file.title}"  )
	except exceptions:
		print( f"Some other YouTube error with song: {dl_file.title}" )




async def SearchW_API( Query, YouTube ):
	search_response = YouTube.search().list(
		q=f'{ Query }',
		part='snippet',
		maxResults=1
	).execute()
	
	for search_result in search_response.get('items', []):
		id = search_result[ 'id' ][ 'videoId' ]

		print( f"Video ID: { id }" )
		# print(f"Title: {search_result['snippet']['title']}")
		# print(f"Description: {search_result['snippet']['description']}")
	
	return id



async def Search2( Query ):	
	try:

		videosSearch = VideosSearch( f'{ Query }', limit = 1 )
		videosResult = await videosSearch.next()

		name = videosResult[ 'result' ][ 0 ][ 'title' ]

		return videosResult[ 'result' ][ 0 ][ 'link' ]
	except IndexError:
		print( "Failed to grab video" )

	# this is a dictionary inside of a list inside of a dictionary

'''
{
	'result': 
	[
		{
			'type': 'video', 
			'id': 'oM2345etj6c', 
			'title': "C'est beau de faire un Show", 
			'publishedTime': None, 
			'duration': '1:40', 
			'viewCount': {
				'text': '2,008 views', 
				'short': '2K views'
			}, 'thumbnails': [{'url': 'https://i.ytimg.com/vi/oM2345etj6c/hq720.jpg?sqp=-oaymwEcCOgCEMoBSFXyq4qpAw4IARUAAIhCGAFwAcABBg==&rs=AOn4CLCQNg6CjxFWe-J-nMx06n3v6K11pQ', 'width': 360, 'height': 202}, {'url': 'https://i.ytimg.com/vi/oM2345etj6c/hq720.jpg?sqp=-oaymwEcCNAFEJQDSFXyq4qpAw4IARUAAIhCGAFwAcABBg==&rs=AOn4CLCajKB8SyosIjqr09GUT6GQCu66Bw', 'width': 720, 'height': 404}], 
			'richThumbnail': None, '
			descriptionSnippet': [{'text': 'Provided to YouTube by Parlophone (France) '}, {'text': "C'est beau de faire un Show", 'bold': True}, {'text': ' · '}, {'text': 'Henri Salvador', 'bold': True}, {'text': " 20 chansons d'or ℗ 1969 Parlophone\xa0..."}], 
			'channel': {'name': 'Henri Salvador - Topic', 'id': 'UCyf5YLGxk0qzOmJzEQqLoFg', 'thumbnails': [{'url': 'https://yt3.ggpht.com/vPT5QGX_VBicj93QAED9qTTmojCOrcxc45xcbWRJk_IdZ6MV-onvYRYBnz8ZFEodum0UACLN=s68-c-k-c0x00ffffff-no-rj', 'width': 68, 'height': 68}], 'link': 'https://www.youtube.com/channel/UCyf5YLGxk0qzOmJzEQqLoFg'}, 
			'accessibility': {'title': "C'est beau de faire un Show by Henri Salvador - Topic 1 minute, 40 seconds 2,008 views", 'duration': '1 minute, 40 seconds'}, 
			'link': 'https://www.youtube.com/watch?v=oM2345etj6c', 
			'shelfTitle': None
		}
	]
}
'''




async def SpotifyFeatures():
	count = 0
	runs = 0

	# with open( "SpotifyFeatures.csv", "r", encoding='utf-8-sig' ) as csv_file:
	with open( "SpotifyFeaturesClean.csv", "r", encoding='utf-8' ) as csv_file:
		reader = csv.reader( csv_file )
		next( reader )

		tracks = []

		for row in reader:
			track = Track( *row )
			tracks.append( track )
			count += 1

	print( count )

	df = pd.read_csv( "SpotifyFeatures.csv", dtype={"genre" : "string", "artist_name" : "string", 
                                                       "track_name" : "string", "track_id" : "string",
                                                       "popularity" : float, "acousticness" : float,
                                                       "danceability" : float, "duration_ms" : int,
                                                       "energy" : float, "instrumentalness" : float,
                                                       "key" : "string", "liveness" : float,
                                                       "loudness" : float, "mode" : "string",
                                                       "speechiness" : float, "tempo" : float,
                                                       "time_signature" : "string", "valence" : float,
                                                       "data" : "string"}, encoding="utf-8")
	

	df["track_name"][0]

	open( 'searching.txt', 'w', encoding='utf-8' ).close

	# with open( "searching.txt", "w", encoding='utf-8' ) as trackfile:
	# 	for i in range(0, len(df["track_name"])):
	# 		df["track_name"][i]
	# 		trackfile.write( f'{ df["track_name"][i] }\n' )

	
	with open( 'searching.txt', 'a', encoding='utf-8' ) as trackfile:
		with open( "songnames.txt", "w", encoding='utf-8' ) as songnames:
			for track in tracks:
				trackfile.write( f'{track}\n' )
				songnames.write( f'{track.name()}\n' )

			print( f"{ track }" )

	# youtube = build('youtube', 'v3', developerKey='not leaking this again')

	with open( 'searching.txt', 'r', encoding='utf-8' ) as trackfile:
		open( 'IDs.txt', 'w', encoding='utf-8' ).close()

		with open( "IDs.txt", "a" ) as idfile:
			with open( "songnames.txt", "r", encoding='utf-8' ) as songnames:
				lines = trackfile.readlines()
				# names = songnames.readlines()

				searches = 0

				for line in lines:

					response = await Search2( Query=line )

					idfile.write( f'{ response }\n' )

					await HandleMP3( response, songnames.readline() )
					
					searches += 1

					if( searches == 10 ):
						converted = audioMod
						converted.batch_convert()
						# break;
						searches = 0
						runs += 1
					
					if( runs == 20 ):
						break;


async def YouTubePlaylist( PlaylistID ):
	YTPlayList = Playlist( PlaylistID )

	YTPlayList._video_regex = re.compile( r"\"url\":\"(/watch\?v=[\w-]*)" )
	
	print( len( YTPlayList.video_urls ) )

	for link in YTPlayList.video_urls:
		try:
			print( link )
			await HandleYTMP3( str( link ) )
		except TypeError:
			print( "Some regex search error somewhere" )


async def SpotifyPlaylist( PlayList ):
	pass

if __name__ == '__main__':

	input_type = int( input( "Type any number to continue\n1. YouTube Playlist\n2. Spotify Playlist\n3. Use SpotifyFeatures.csv\n>> " ) )

	if( input_type == 1 ):
		YT_List = input( "YouTube Playlist URL: " )
		asyncio.run( YouTubePlaylist( YT_List ) )
	elif( input_type == 2 ):
		Spot_List = input( "Spotify Playlist URL: " )
		asyncio.run( SpotifyPlaylist( Spot_List ) )
	elif( input_type == 3 ):
		try:
			asyncio.run( SpotifyFeatures() )
		except TypeError:
			print( "Error with SpotifyFeatures, most likely a regex search error" )


