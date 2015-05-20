package mllib.perf.clustering

import mllib.perf.util.DataGenerator
import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.{PowerIterationClusteringModel, PowerIterationClustering}
import org.apache.spark.rdd.RDD

import mllib.perf.{PerfTest}
import collection.mutable
import org.json4s.JValue
import org.json4s.JsonDSL._

class PowerIterationClusteringTest(sc: SparkContext) extends PerfTest {

  type RDDType = (Long, Long, Double)
  // TODO: refactor k-means and GMM code
  val NUM_POINTS = ("num-points", "number of points for clustering tests")
  val NUM_CLUSTERS = ("num-clusters", "number of centers for clustering tests")
//  val NUM_ITERATIONS = ("num-iterations", "number of iterations for the algorithm")
  val WEAK_LINK_FACTOR = ("weak-link-factor", "link weight for links that represent graph partitions")

  intOptions ++= Seq(NUM_CLUSTERS /* NUM_ITERATIONS*/)
  longOptions ++= Seq(NUM_POINTS)
  doubleOptions ++= Seq(WEAK_LINK_FACTOR)
  val options = intOptions ++ stringOptions  ++ booleanOptions ++ longOptions ++ doubleOptions
  addOptionsToParser()

  def runTest(rdd: RDD[RDDType]): PowerIterationClusteringModel = {

    val nWeakLinks: Int = intOptionValue(NUM_CLUSTERS)
    val similarities = rdd
    val model = new PowerIterationClustering()
      .setK(nWeakLinks)
      .run(similarities)
    model
  }

  var similarities: RDD[RDDType] = _
  var weakIndices: Seq[Long] = _

  override def createInputData(seed: Long): Unit = {
    val nVertices = longOptionValue(NUM_POINTS).toInt
    System.err.println(s"nVertices=$nVertices")
    val nWeakLinks: Int = intOptionValue(NUM_CLUSTERS)
    val nPartitions = intOptionValue(NUM_PARTITIONS)
    val weakLinkFactor = doubleOptionValue(WEAK_LINK_FACTOR)
    val (similaritiesTmp, weakIndicesTmp) =
      DataGenerator.generateSinglyConnectedGraphWithWeakLinks(sc,
        nVertices, nWeakLinks, nPartitions, weakLinkFactor)
    similarities = similaritiesTmp
    weakIndices = weakIndicesTmp
  }

  override def run(): JValue = {
    val start = System.currentTimeMillis
    val model = runTest(similarities)
    val nVertices: Int = longOptionValue(NUM_POINTS).toInt
    var weakx = 0
    val expected = Array.fill(nVertices)(0L)
    var wwx = 0
    for (wx <- weakIndices) {
      wwx += 1
      var px = 0
      while (px < wx && px < nVertices) {
        expected(px) = wwx
        px += 1
      }
      if (px < nVertices) {
        expected(px) = -1L
      }
    }
    val predictions = mutable.ArrayBuffer[Long](nVertices)
    model.assignments.collect().foreach { a =>
      predictions += a.id
    }
    System.err.println(s"predictions = ${predictions.mkString(",")}")
    System.err.println(s"expected = ${expected.mkString(",")}")
    assert(predictions == expected)

    val duration = (System.currentTimeMillis() - start) / 1e3
    "time" -> duration
  }


}
