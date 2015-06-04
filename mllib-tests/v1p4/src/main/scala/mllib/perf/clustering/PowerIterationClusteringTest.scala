package mllib.perf.clustering

import mllib.perf.util.DataGenerator
import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.{PowerIterationClusteringModel, PowerIterationClustering}
import org.apache.spark.rdd.RDD

import mllib.perf.{PerfTest}
import collection.mutable
import org.json4s.JValue
import org.json4s.JsonDSL._
import java.util.{TreeMap => JTreeMap}

class PowerIterationClusteringTest(sc: SparkContext) extends PerfTest {

  type RDDType = (Long, Long, Double)
  val NUM_POINTS = ("num-points", "number of points for clustering tests")
  val NUM_CLUSTERS = ("num-clusters", "number of clusters")
  val WEAK_LINK_FACTOR = ("weak-link-factor", "link weight for links that represent graph partitions")

  intOptions ++= Seq(NUM_CLUSTERS)
  longOptions ++= Seq(NUM_POINTS)
  doubleOptions ++= Seq(WEAK_LINK_FACTOR)
  val options = intOptions ++ stringOptions ++ booleanOptions ++ longOptions ++ doubleOptions
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
    val expected = new JTreeMap[Int, mutable.TreeSet[Long]]()
    var wwx = 0
    var px = 1
    for (wx <- mutable.Seq[Long](weakIndices: _*) :+ nVertices.toLong) {
      val set = new mutable.TreeSet[Long]()
      expected.put(wwx, set)
      while (px <= wx.asInstanceOf[Long] && px <= nVertices) {
        set.add(px)
        px += 1
      }
      wwx += 1
    }
    val predictions = new JTreeMap[Int, mutable.TreeSet[Long]]()
    model.assignments.collect().foreach { a =>
      var set = predictions.get(a.cluster)
      if (set == null) {
        set = new mutable.TreeSet[Long]()
        predictions.put(a.cluster, set)
      }
      set.add(a.id)
    }
    import collection.JavaConverters._
    val smap = mutable.TreeSet(predictions.keySet.asScala.toSeq: _*).map { k =>
      s"$k:${predictions.get(k).mkString(",")}"
    }
    System.err.println(s"weaklinks=${weakIndices.mkString(",")}")
    System.err.println(s"predictions = ${smap.mkString("{", " ; ", "}")}")
    val emap = mutable.TreeSet(expected.keySet.asScala.toSeq: _*).map { k =>
      s"$k:${expected.get(k).mkString(",")}"
    }
    System.err.println(s"expected = ${emap.mkString("{", " ; ", "}")}")
    assert(smap == emap)

    val duration = (System.currentTimeMillis() - start) / 1e3
    "time" -> duration
  }

}
